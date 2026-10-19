"""
Milestone 1 Test Suite: Start/Sit Lineup Optimizer & Schedule Engine.

Tests 100% offline deterministic functionality:
1. ScheduleCalendar (Schedule Engine, synthetic generator, lookups, streamers).
2. PointsScoringConfig (Weekly lock mode validation).
3. Start/Sit Lineup Optimizer (Daily bipartite matching, weekly volume optimization,
   locking validation, IR/bench segregation, multi-position cascading, dict backward compatibility).
"""

import pytest
import numpy as np
import pandas as pd

from backend.points.schedule import (
    ScheduleCalendar,
    generate_synthetic_nba_schedule,
    normalize_team,
    normalize_date,
    date_range,
    NBAGame,
    NBA_TEAMS,
)
from backend.points.config import (
    build_points_scoring_config,
    PointsConfigError,
)
from backend.points.lineups import (
    DailyLineupAssignment,
    WeeklyLineupAssignment,
    assign_optimal_daily_lineup,
    assign_optimal_weekly_lineup,
    compute_lineup_fp,
    build_slot_configs_from_position_config,
    DEFAULT_POSITION_STRUCTURE,
)
from backend.points.roster_schedule import (
    compute_marginal_lineup_delta_fp,
    compute_marginal_lineup_detailed,
    compute_multi_day_lineup_fp,
    evaluate_waiver_candidates,
    WaiverValuationResult,
)
from backend.points.streamer import (
    optimize_streaming_acquisitions,
    analyze_schedule_density,
    StreamingMove,
    StreamerSchedulePlan,
    StreamerOptimizationResult,
)


# ==============================================================================
# 1. Schedule Engine & Multi-Day Calendar Tests
# ==============================================================================

class TestScheduleEngine:
    """Tests for ScheduleCalendar and deterministic synthetic schedule generation."""

    def test_schedule_deterministic_fallback(self):
        """Verify default instantiation produces deterministic 30-team schedule."""
        cal1 = ScheduleCalendar()
        cal2 = ScheduleCalendar()

        # All 30 teams exist
        for t in NBA_TEAMS:
            dates1 = cal1.get_team_game_dates(t, "2024-10-21", "2025-04-13")
            dates2 = cal2.get_team_game_dates(t, "2024-10-21", "2025-04-13")
            assert len(dates1) == 85
            assert dates1 == dates2  # Determinism check

    def test_schedule_team_game_dates_filtering_and_aliases(self):
        """Verify date window filtering, alias mapping, and inverted window handling."""
        cal = ScheduleCalendar()

        # Normal query
        bos_dates = cal.get_team_game_dates("BOS", "2024-10-21", "2024-10-27")
        assert all("2024-10-21" <= d <= "2024-10-27" for d in bos_dates)

        # Alias resolution ('GS' -> 'GSW', 'BRK' -> 'BKN')
        gsw_dates1 = cal.get_team_game_dates("GSW", "2024-10-21", "2024-10-27")
        gsw_dates2 = cal.get_team_game_dates("GS", "2024-10-21", "2024-10-27")
        assert gsw_dates1 == gsw_dates2

        # Inverted date window returns empty list
        assert cal.get_team_game_dates("BOS", "2024-10-27", "2024-10-21") == []

        # Unknown team returns empty list
        assert cal.get_team_game_dates("NONEXISTENT", "2024-10-21", "2024-10-27") == []

    def test_schedule_player_game_dates(self):
        """Verify player-to-team registry mapping and player game date lookups."""
        cal = ScheduleCalendar()
        cal.register_player_team(101, "LAL")
        cal.register_player_team(102, "BOS")

        lal_dates = cal.get_team_game_dates("LAL", "2024-10-21", "2024-10-27")
        assert cal.get_player_game_dates(101, "2024-10-21", "2024-10-27") == lal_dates

        # Unregistered player returns []
        assert cal.get_player_game_dates(999, "2024-10-21", "2024-10-27") == []

        # Player removal
        cal.remove_player(101)
        assert cal.get_player_game_dates(101, "2024-10-21", "2024-10-27") == []

    def test_schedule_get_players_playing_on_date(self):
        """Verify get_players_playing_on_date filters down to exact players active on date."""
        cal = ScheduleCalendar.from_dict({
            "BOS": ["2024-11-01", "2024-11-03"],
            "NYK": ["2024-11-01", "2024-11-04"],
            "MIA": ["2024-11-02", "2024-11-04"],
        }, player_teams={1: "BOS", 2: "NYK", 3: "MIA", 4: "LAL"})

        active_nov1 = cal.get_players_playing_on_date([1, 2, 3, 4], "2024-11-01")
        assert active_nov1 == {1, 2}

        active_nov2 = cal.get_players_playing_on_date([1, 2, 3, 4], "2024-11-02")
        assert active_nov2 == {3}

        # Date with no games
        assert cal.get_players_playing_on_date([1, 2, 3, 4], "2024-11-10") == set()

    def test_schedule_get_roster_games_in_window(self):
        """Verify get_roster_games_in_window returns exact game counts per player."""
        cal = ScheduleCalendar.from_dict({
            "BOS": ["2024-10-21", "2024-10-23", "2024-10-25", "2024-10-27"],
            "LAL": ["2024-10-22", "2024-10-24", "2024-10-26"],
            "CHI": ["2024-10-21", "2024-10-26"],
        }, player_teams={10: "BOS", 20: "LAL", 30: "CHI", 40: "MIA"})

        counts = cal.get_roster_games_in_window([10, 20, 30, 40], "2024-10-21", "2024-10-27")
        assert counts == {10: 4, 20: 3, 30: 2, 40: 0}

    def test_schedule_empty_calendar(self):
        """Verify empty ScheduleCalendar safely handles all query methods."""
        cal = ScheduleCalendar(team_schedules={})
        assert cal.get_team_game_dates("BOS", "2024-10-21", "2024-10-27") == []
        assert cal.get_player_game_dates(1, "2024-10-21", "2024-10-27") == []
        assert cal.get_players_playing_on_date([1, 2], "2024-10-21") == set()
        assert cal.get_roster_games_in_window([1, 2], "2024-10-21", "2024-10-27") == {1: 0, 2: 0}
        assert not cal.is_team_playing_on_date("BOS", "2024-10-21")
        assert not cal.is_player_playing_on_date(1, "2024-10-21")

    def test_schedule_single_day_window(self):
        """Verify behavior when start_date equals end_date."""
        cal = ScheduleCalendar.from_dict({
            "DEN": ["2024-11-15"]
        })
        assert cal.get_team_game_count("DEN", "2024-11-15", "2024-11-15") == 1
        assert cal.get_team_game_count("DEN", "2024-11-16", "2024-11-16") == 0

    def test_schedule_string_vs_int_player_ids(self):
        """Verify string player IDs are coerced to ints on registration."""
        cal = ScheduleCalendar.from_dict(
            {"GSW": ["2024-10-23"]},
            player_teams={"1001": "GSW"}
        )
        assert cal.is_player_playing_on_date(1001, "2024-10-23")
        assert cal.get_player_team(1001) == "GSW"

    def test_schedule_window_normalization(self):
        """Verify normalize_window supports tuple, list, str, and None."""
        cal = ScheduleCalendar()
        # Tuple
        w_tuple = cal.normalize_window(("2024-10-21", "2024-10-23"))
        assert w_tuple == ["2024-10-21", "2024-10-22", "2024-10-23"]

        # List
        w_list = cal.normalize_window(["2024-10-23", "2024-10-21"])
        assert w_list == ["2024-10-21", "2024-10-23"]

        # Single string
        w_str = cal.normalize_window("2024-10-21")
        assert w_str == ["2024-10-21"]

    def test_schedule_back_to_back_detection(self):
        """Verify get_team_back_to_backs correctly identifies consecutive game days."""
        cal = ScheduleCalendar.from_dict({
            "MIL": ["2024-11-01", "2024-11-02", "2024-11-04", "2024-11-05", "2024-11-07"]
        })
        b2b = cal.get_team_back_to_backs("MIL", "2024-11-01", "2024-11-07")
        assert b2b == [("2024-11-01", "2024-11-02"), ("2024-11-04", "2024-11-05")]

    def test_schedule_density_3_in_4(self):
        """Verify get_games_density calculates maximum games in rolling span."""
        cal = ScheduleCalendar.from_dict({
            "PHX": ["2024-11-01", "2024-11-02", "2024-11-04", "2024-11-08"]
        })
        # 3 games in 4 days (Nov 1, 2, 4)
        density_4 = cal.get_games_density("PHX", "2024-11-01", "2024-11-10", span_days=4)
        assert density_4 == 3

    def test_schedule_light_slate_dates(self):
        """Verify light slate identification based on team counts."""
        cal = ScheduleCalendar.from_dict({
            "BOS": ["2024-11-01", "2024-11-02"],
            "NYK": ["2024-11-01", "2024-11-02"],
            "MIA": ["2024-11-01"],
            "CHI": ["2024-11-01"],
        })
        # Nov 1 has 4 teams, Nov 2 has 2 teams
        light = cal.get_light_slate_dates(["2024-11-01", "2024-11-02"], max_teams=2)
        assert light == ["2024-11-02"]

    def test_schedule_no_three_in_a_row_constraint(self):
        """Verify synthetic NBA schedule generator enforces strictly NO 3-in-3 games."""
        sched, _ = generate_synthetic_nba_schedule(start_date="2024-10-21", weeks=25, seed=123)
        for team, dates in sched.items():
            dt_set = {normalize_date(d) for d in dates}
            for d_str in dates:
                dt = normalize_date(d_str)
                d_obj = pd.to_datetime(dt).date()
                d_minus_1 = (d_obj - pd.Timedelta(days=1)).isoformat()
                d_minus_2 = (d_obj - pd.Timedelta(days=2)).isoformat()
                assert not (d_minus_1 in dt_set and d_minus_2 in dt_set), (
                    f"Team {team} violated 3-in-a-row constraint on {dt}"
                )

    def test_schedule_from_dataframe(self):
        """Verify loading ScheduleCalendar from a pandas DataFrame."""
        df = pd.DataFrame([
            {"date": "2024-11-01", "home_team": "BOS", "away_team": "MIA"},
            {"date": "2024-11-03", "home_team": "LAL", "away_team": "GSW"},
        ])
        cal = ScheduleCalendar.from_dataframe(df)
        assert cal.is_team_playing_on_date("BOS", "2024-11-01")
        assert cal.is_team_playing_on_date("MIA", "2024-11-01")
        assert cal.is_team_playing_on_date("LAL", "2024-11-03")
        assert not cal.is_team_playing_on_date("BOS", "2024-11-02")

    def test_schedule_from_snowflake_offline_fallback(self):
        """Verify ScheduleCalendar.from_snowflake gracefully falls back to synthetic when offline."""
        cal = ScheduleCalendar.from_snowflake(fallback_to_synthetic=True)
        assert len(cal._team_schedules) == 30


# ==============================================================================
# 2. Points Config Tests
# ==============================================================================

class TestPointsConfig:
    """Tests for PointsScoringConfig lock mode validation."""

    def test_config_weekly_lock_mode_validation(self):
        """Verify build_points_scoring_config accepts lineup_lock_mode='weekly'."""
        cfg = build_points_scoring_config(
            points_input_mode="projected_points",
            projection_basis="per_game",
            scoring_rules={},
            lineup_lock_mode="weekly",
        )
        assert cfg.lineup_lock_mode == "weekly"

    def test_config_pregame_daily_lock_mode_validation(self):
        """Verify build_points_scoring_config accepts lineup_lock_mode='pregame_daily'."""
        cfg = build_points_scoring_config(
            points_input_mode="projected_points",
            projection_basis="per_game",
            scoring_rules={},
            lineup_lock_mode="pregame_daily",
        )
        assert cfg.lineup_lock_mode == "pregame_daily"

    def test_config_invalid_lock_mode_rejected(self):
        """Verify invalid lineup_lock_mode raises PointsConfigError."""
        with pytest.raises(PointsConfigError, match="Invalid value 'monthly'"):
            build_points_scoring_config(
                points_input_mode="projected_points",
                projection_basis="per_game",
                scoring_rules={},
                lineup_lock_mode="monthly",
            )


# ==============================================================================
# 3. Lineup Optimizer Tests (Daily & Weekly)
# ==============================================================================

class TestLineupOptimizer:
    """Comprehensive test suite covering daily & weekly start/sit optimization."""

    # --------------------------------------------------------------------------
    # M1-T01: Daily optimal assignment basic
    # --------------------------------------------------------------------------
    def test_daily_optimal_assignment_basic(self):
        """10 active slots, 10 active players: 100% legal matching, exact FP sum."""
        roster = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        positions = {
            1: ["PG"], 2: ["SG"], 3: ["SF"], 4: ["PF"], 5: ["C"],
            6: ["PG"], 7: ["SG"], 8: ["SF"], 9: ["PF"], 10: ["C"],
        }
        fp = {i: 30.0 for i in roster}
        slots = {"PG": 2, "SG": 2, "SF": 2, "PF": 2, "C": 2}

        assignment = assign_optimal_daily_lineup(
            roster_player_ids=roster,
            player_positions=positions,
            player_fp=fp,
            active_slots=slots,
        )

        assert len(assignment.active_assignments) == 10
        assert all(pid is not None for pid in assignment.active_assignments.values())
        assert assignment.total_expected_fp == 300.0
        assert assignment.bench_players == []
        assert assignment.ir_players == []

    # --------------------------------------------------------------------------
    # M1-T02: Daily displacement of lower scorers
    # --------------------------------------------------------------------------
    def test_daily_displacement_lower_scorers(self):
        """13 players for 10 slots: 3 lowest scorers benched, top 10 start."""
        roster = list(range(1, 14))
        # All Util eligible for clean displacement test
        positions = {i: ["PG", "SG", "SF", "PF", "C"] for i in roster}
        # Player 1..10 have FP 30..39; Players 11, 12, 13 have FP 10, 12, 15
        fp = {i: float(20 + i) for i in range(1, 11)}
        fp[11] = 10.0
        fp[12] = 12.0
        fp[13] = 15.0

        slots = {"Util": 10}

        assignment = assign_optimal_daily_lineup(
            roster_player_ids=roster,
            player_positions=positions,
            player_fp=fp,
            active_slots=slots,
        )

        starters = set(assignment.active_assignments.values())
        assert starters == set(range(1, 11))
        assert set(assignment.bench_players) == {11, 12, 13}
        assert assignment.total_expected_fp == sum(fp[i] for i in range(1, 11))

    # --------------------------------------------------------------------------
    # M1-T03: Multi-position flex cascading
    # --------------------------------------------------------------------------
    def test_daily_multi_position_flex_cascading(self):
        """PG/SG moves to SG slot so higher-scoring pure PG can start."""
        # Player 1: PG (40.0 FP)
        # Player 2: PG/SG (35.0 FP)
        # Player 3: SG (20.0 FP)
        roster = [1, 2, 3]
        positions = {1: ["PG"], 2: ["PG", "SG"], 3: ["SG"]}
        fp = {1: 40.0, 2: 35.0, 3: 20.0}
        slots = {"PG": 1, "SG": 1}

        assignment = assign_optimal_daily_lineup(
            roster_player_ids=roster,
            player_positions=positions,
            player_fp=fp,
            active_slots=slots,
        )

        assert assignment["PG1"] == 1
        assert assignment["SG1"] == 2
        assert assignment.bench_players == [3]
        assert assignment.total_expected_fp == 75.0

    # --------------------------------------------------------------------------
    # M1-T04: 3-way domino flex chain
    # --------------------------------------------------------------------------
    def test_daily_domino_flex_chain_3way(self):
        """PG -> PG/SG -> SG/SF -> pure SF domino flow."""
        # 1: pure PG (50 FP)
        # 2: PG/SG (45 FP)
        # 3: SG/SF (40 FP)
        # 4: pure SF (25 FP)
        roster = [1, 2, 3, 4]
        positions = {1: ["PG"], 2: ["PG", "SG"], 3: ["SG", "SF"], 4: ["SF"]}
        fp = {1: 50.0, 2: 45.0, 3: 40.0, 4: 25.0}
        slots = {"PG": 1, "SG": 1, "SF": 1}

        assignment = assign_optimal_daily_lineup(
            roster_player_ids=roster,
            player_positions=positions,
            player_fp=fp,
            active_slots=slots,
        )

        assert assignment["PG1"] == 1
        assert assignment["SG1"] == 2
        assert assignment["SF1"] == 3
        assert assignment.bench_players == [4]
        assert assignment.total_expected_fp == 135.0

    # --------------------------------------------------------------------------
    # M1-T05: Daily off-day schedule filter
    # --------------------------------------------------------------------------
    def test_daily_off_day_schedule_filter(self):
        """playing_today set restricts starters; off-day players benched."""
        roster = [1, 2, 3]
        positions = {1: ["PG"], 2: ["SG"], 3: ["PG"]}
        fp = {1: 50.0, 2: 30.0, 3: 25.0}
        slots = {"PG": 1, "SG": 1}

        # Player 1 (50 FP) is OFF today
        playing_today = {2, 3}

        assignment = assign_optimal_daily_lineup(
            roster_player_ids=roster,
            player_positions=positions,
            player_fp=fp,
            active_slots=slots,
            playing_today=playing_today,
        )

        assert assignment["PG1"] == 3
        assert assignment["SG1"] == 2
        assert assignment.bench_players == [1]
        assert assignment.total_expected_fp == 55.0

    # --------------------------------------------------------------------------
    # M1-T06: Daily negative FP player benched
    # --------------------------------------------------------------------------
    def test_daily_negative_fp_player_benched(self):
        """Scrub with -5.0 FP left benched rather than filling open Util slot."""
        roster = [1, 2]
        positions = {1: ["PG"], 2: ["PG"]}
        fp = {1: 30.0, 2: -5.0}
        slots = {"PG": 1, "Util": 1}

        assignment = assign_optimal_daily_lineup(
            roster_player_ids=roster,
            player_positions=positions,
            player_fp=fp,
            active_slots=slots,
        )

        assert assignment["PG1"] == 1
        assert assignment["Util1"] is None
        assert assignment.bench_players == [2]
        assert assignment.total_expected_fp == 30.0

    # --------------------------------------------------------------------------
    # M1-T07: Daily zero FP player handling
    # --------------------------------------------------------------------------
    def test_daily_zero_fp_player_handling(self):
        """Player with 0.0 projected FP handled cleanly without crash."""
        roster = [1, 2]
        positions = {1: ["PG"], 2: ["SG"]}
        fp = {1: 30.0, 2: 0.0}
        slots = {"PG": 1, "SG": 1}

        assignment = assign_optimal_daily_lineup(
            roster_player_ids=roster,
            player_positions=positions,
            player_fp=fp,
            active_slots=slots,
        )

        assert assignment["PG1"] == 1
        # Eligible 0.0 FP player fills slot
        assert assignment["SG1"] == 2
        assert assignment.total_expected_fp == 30.0

    # --------------------------------------------------------------------------
    # M1-T08: Weekly volume displacement 3g vs 4g
    # --------------------------------------------------------------------------
    def test_weekly_volume_displacement_3g_vs_4g(self):
        """4-game player (26 FP/G = 104 FP) starts over 3-game player (32 FP/G = 96 FP)."""
        roster = [1, 2]
        positions = {1: ["Util"], 2: ["Util"]}
        fpg = {1: 32.0, 2: 26.0}
        games = {1: 3, 2: 4}
        # W1 = 96.0, W2 = 104.0
        slots = {"Util": 1}

        assignment = assign_optimal_weekly_lineup(
            roster_player_ids=roster,
            player_positions=positions,
            player_fp_per_game=fpg,
            player_games_in_week=games,
            active_slots=slots,
        )

        assert assignment["Util1"] == 2
        assert assignment.bench_players == [1]
        assert assignment.total_expected_fp == 104.0

    # --------------------------------------------------------------------------
    # M1-T09: Weekly superstar volume invariance
    # --------------------------------------------------------------------------
    def test_weekly_superstar_volume_invariance(self):
        """3-game superstar (48 FP/G = 144 FP) starts over 5-game player (27 FP/G = 135 FP)."""
        roster = [1, 2]
        positions = {1: ["Util"], 2: ["Util"]}
        fpg = {1: 48.0, 2: 27.0}
        games = {1: 3, 2: 5}
        # W1 = 144.0, W2 = 135.0
        slots = {"Util": 1}

        assignment = assign_optimal_weekly_lineup(
            roster_player_ids=roster,
            player_positions=positions,
            player_fp_per_game=fpg,
            player_games_in_week=games,
            active_slots=slots,
        )

        assert assignment["Util1"] == 1
        assert assignment.bench_players == [2]
        assert assignment.total_expected_fp == 144.0

    # --------------------------------------------------------------------------
    # M1-T10: Weekly multi-position flex assignment
    # --------------------------------------------------------------------------
    def test_weekly_multi_position_flex_assignment(self):
        """Multi-position optimal weekly allocation under W_i = FP/G_i * Games_i."""
        roster = [1, 2, 3, 4]
        positions = {1: ["PG"], 2: ["PG", "SG"], 3: ["PG"], 4: ["SG"]}
        fpg = {1: 35.0, 2: 30.0, 3: 42.0, 4: 25.0}
        games = {1: 4, 2: 4, 3: 2, 4: 3}
        # W: 1: 140, 2: 120, 3: 84, 4: 75
        slots = {"PG": 1, "SG": 1, "Util": 1}

        assignment = assign_optimal_weekly_lineup(
            roster_player_ids=roster,
            player_positions=positions,
            player_fp_per_game=fpg,
            player_games_in_week=games,
            active_slots=slots,
        )

        assert assignment["PG1"] == 1
        assert assignment["SG1"] == 2
        assert assignment["Util1"] == 3
        assert assignment.bench_players == [4]
        assert assignment.total_expected_fp == 344.0

    # --------------------------------------------------------------------------
    # M1-T11: Weekly zero-game player benched
    # --------------------------------------------------------------------------
    def test_weekly_zero_game_player_benched(self):
        """Player with 0 games in week placed on bench."""
        roster = [1, 2]
        positions = {1: ["Util"], 2: ["Util"]}
        fpg = {1: 40.0, 2: 25.0}
        games = {1: 0, 2: 3}
        slots = {"Util": 1}

        assignment = assign_optimal_weekly_lineup(
            roster_player_ids=roster,
            player_positions=positions,
            player_fp_per_game=fpg,
            player_games_in_week=games,
            active_slots=slots,
        )

        assert assignment["Util1"] == 2
        assert 1 in assignment.bench_players or 1 in assignment.ir_players
        assert assignment.total_expected_fp == 75.0

    # --------------------------------------------------------------------------
    # M1-T12: Locking single active slot
    # --------------------------------------------------------------------------
    def test_locking_single_active_slot(self):
        """Locked player preserved in slot; rest of roster optimized around them."""
        roster = [1, 2, 3]
        positions = {1: ["PG"], 2: ["PG"], 3: ["SG"]}
        fp = {1: 40.0, 2: 30.0, 3: 25.0}
        slots = {"PG": 1, "SG": 1}

        # User locks player 2 (30.0 FP) into PG1
        assignment = assign_optimal_daily_lineup(
            roster_player_ids=roster,
            player_positions=positions,
            player_fp=fp,
            active_slots=slots,
            locked_assignments={"PG1": 2},
        )

        assert assignment["PG1"] == 2
        assert assignment["SG1"] == 3
        assert assignment.bench_players == [1]  # Higher scorer 1 forced to bench
        assert assignment.total_expected_fp == 55.0

    # --------------------------------------------------------------------------
    # M1-T13: Locking suboptimal player displaces starter
    # --------------------------------------------------------------------------
    def test_locking_suboptimal_player_displaces_starter(self):
        """User locks low scorer; higher scorer benched; total FP is strictly lower."""
        roster = [1, 2]
        positions = {1: ["Util"], 2: ["Util"]}
        fp = {1: 45.0, 2: 15.0}
        slots = {"Util": 1}

        # Unlocked: Player 1 starts
        unlocked = assign_optimal_daily_lineup(
            roster_player_ids=roster, player_positions=positions, player_fp=fp, active_slots=slots
        )
        assert unlocked["Util1"] == 1
        assert unlocked.total_expected_fp == 45.0

        # Locked to suboptimal Player 2
        locked = assign_optimal_daily_lineup(
            roster_player_ids=roster,
            player_positions=positions,
            player_fp=fp,
            active_slots=slots,
            locked_assignments={"Util1": 2},
        )
        assert locked["Util1"] == 2
        assert locked.total_expected_fp == 15.0

    # --------------------------------------------------------------------------
    # M1-T14: Locking all active slots short-circuit
    # --------------------------------------------------------------------------
    def test_locking_all_active_slots_short_circuit(self):
        """Fully pre-locked roster short-circuits solver with exact assignments preserved."""
        roster = [1, 2]
        positions = {1: ["PG"], 2: ["SG"]}
        fp = {1: 30.0, 2: 25.0}
        slots = {"PG": 1, "SG": 1}

        assignment = assign_optimal_daily_lineup(
            roster_player_ids=roster,
            player_positions=positions,
            player_fp=fp,
            active_slots=slots,
            locked_assignments={"PG1": 1, "SG1": 2},
        )

        assert assignment["PG1"] == 1
        assert assignment["SG1"] == 2
        assert assignment.total_expected_fp == 55.0

    # --------------------------------------------------------------------------
    # M1-T15: Locking duplicate player raises error
    # --------------------------------------------------------------------------
    def test_locking_duplicate_player_raises_error(self):
        """Locking same player to multiple slots raises ValueError."""
        roster = [1, 2]
        positions = {1: ["PG", "SG"], 2: ["Util"]}
        fp = {1: 30.0, 2: 20.0}
        slots = {"PG": 1, "SG": 1}

        with pytest.raises(ValueError, match="Duplicate player IDs found"):
            assign_optimal_daily_lineup(
                roster_player_ids=roster,
                player_positions=positions,
                player_fp=fp,
                active_slots=slots,
                locked_assignments={"PG1": 1, "SG1": 1},
            )

    # --------------------------------------------------------------------------
    # M1-T16: Locking player not on roster raises error
    # --------------------------------------------------------------------------
    def test_locking_player_not_on_roster_raises_error(self):
        """Locking player not on roster raises ValueError."""
        roster = [1, 2]
        positions = {1: ["PG"], 2: ["SG"]}
        fp = {1: 30.0, 2: 20.0}
        slots = {"PG": 1, "SG": 1}

        with pytest.raises(ValueError, match="is not on roster"):
            assign_optimal_daily_lineup(
                roster_player_ids=roster,
                player_positions=positions,
                player_fp=fp,
                active_slots=slots,
                locked_assignments={"PG1": 999},
            )

    # --------------------------------------------------------------------------
    # M1-T17: IR slot capacity and overflow
    # --------------------------------------------------------------------------
    def test_ir_slot_capacity_and_overflow(self):
        """4 injured players with ir_slots=3: 3 in IR, 1 overflows to bench."""
        roster = [1, 2, 10, 11, 12, 13]
        positions = {i: ["Util"] for i in roster}
        fp = {1: 40.0, 2: 35.0, 10: np.nan, 11: np.nan, 12: np.nan, 13: np.nan}
        slots = {"Util": 2}
        injured = {10, 11, 12, 13}

        assignment = assign_optimal_daily_lineup(
            roster_player_ids=roster,
            player_positions=positions,
            player_fp=fp,
            active_slots=slots,
            ir_slots=3,
            injured_players=injured,
        )

        assert set(assignment.active_assignments.values()) == {1, 2}
        assert len(assignment.ir_players) == 3
        assert set(assignment.ir_players).issubset(injured)
        # 1 injured player overflowed to bench
        assert len(assignment.bench_players) == 1
        assert assignment.bench_players[0] in injured

    # --------------------------------------------------------------------------
    # M1-T18: Healthy players never in IR
    # --------------------------------------------------------------------------
    def test_healthy_players_never_in_ir(self):
        """Open IR slots remain empty when bench players are healthy."""
        roster = [1, 2, 3, 4]
        positions = {i: ["Util"] for i in roster}
        fp = {1: 40.0, 2: 35.0, 3: 30.0, 4: 25.0}
        slots = {"Util": 2}

        assignment = assign_optimal_daily_lineup(
            roster_player_ids=roster,
            player_positions=positions,
            player_fp=fp,
            active_slots=slots,
            ir_slots=3,
            injured_players=set(),  # All healthy
        )

        assert set(assignment.active_assignments.values()) == {1, 2}
        assert assignment.ir_players == []
        assert set(assignment.bench_players) == {3, 4}

    # --------------------------------------------------------------------------
    # M1-T19: Boundary empty roster
    # --------------------------------------------------------------------------
    def test_boundary_empty_roster(self):
        """Empty roster returns all slots None, 0.0 FP, empty bench/IR."""
        slots = {"PG": 1, "SG": 1}
        assignment = assign_optimal_daily_lineup(
            roster_player_ids=[],
            player_positions={},
            player_fp={},
            active_slots=slots,
        )

        assert assignment["PG1"] is None
        assert assignment["SG1"] is None
        assert assignment.bench_players == []
        assert assignment.ir_players == []
        assert assignment.total_expected_fp == 0.0

    # --------------------------------------------------------------------------
    # M1-T20: Boundary single player
    # --------------------------------------------------------------------------
    def test_boundary_single_player(self):
        """1 player on 10-slot team starts in eligible slot; 9 slots None."""
        roster = [1]
        positions = {1: ["PG"]}
        fp = {1: 35.0}
        slots = {"PG": 1, "SG": 1, "SF": 1, "PF": 1, "C": 1, "Util": 5}

        assignment = assign_optimal_daily_lineup(
            roster_player_ids=roster,
            player_positions=positions,
            player_fp=fp,
            active_slots=slots,
        )

        assert assignment["PG1"] == 1
        assert sum(1 for v in assignment.values() if v is not None) == 1
        assert assignment.total_expected_fp == 35.0

    # --------------------------------------------------------------------------
    # M1-T21: Boundary all players ineligible
    # --------------------------------------------------------------------------
    def test_boundary_all_players_ineligible(self):
        """All players ineligible for available slots; all benched, slots None."""
        roster = [1, 2]
        positions = {1: ["C"], 2: ["C"]}
        fp = {1: 35.0, 2: 30.0}
        slots = {"PG": 1, "SG": 1}

        assignment = assign_optimal_daily_lineup(
            roster_player_ids=roster,
            player_positions=positions,
            player_fp=fp,
            active_slots=slots,
        )

        assert assignment["PG1"] is None
        assert assignment["SG1"] is None
        assert set(assignment.bench_players) == {1, 2}
        assert assignment.total_expected_fp == 0.0

    # --------------------------------------------------------------------------
    # M1-T22: Backward compatibility dict operations
    # --------------------------------------------------------------------------
    def test_backward_compatibility_dict_operations(self):
        """Verify dict mapping methods on DailyLineupAssignment and WeeklyLineupAssignment."""
        roster = [1, 2]
        positions = {1: ["PG"], 2: ["SG"]}
        fp = {1: 30.0, 2: 25.0}
        slots = {"PG": 1, "SG": 1}

        daily = assign_optimal_daily_lineup(roster, positions, fp, slots)

        # Mapping interface
        assert daily["PG1"] == 1
        assert daily.get("PG1") == 1
        assert daily.get("Unknown", None) is None
        assert "PG1" in daily
        assert len(daily) == 2
        assert list(daily.keys()) == ["PG1", "SG1"]
        assert list(daily.values()) == [1, 2]
        assert set(daily.items()) == {("PG1", 1), ("SG1", 2)}
        assert list(iter(daily)) == ["PG1", "SG1"]

        # compute_lineup_fp
        assert compute_lineup_fp(daily, fp) == 55.0
        assert compute_lineup_fp(daily) == 55.0

        # Weekly
        games = {1: 3, 2: 4}
        weekly = assign_optimal_weekly_lineup(roster, positions, fp, games, slots)
        assert weekly["PG1"] == 1
        assert weekly.get("SG1") == 2
        assert len(weekly) == 2
        assert compute_lineup_fp(weekly) == 190.0

    # --------------------------------------------------------------------------
    # M1-T23: Backward compatibility legacy call signature
    # --------------------------------------------------------------------------
    def test_backward_compatibility_legacy_signature(self):
        """Verify assign_optimal_daily_lineup works with legacy keyword arguments."""
        roster = [1, 2]
        per_game_fp = {1: 35.0, 2: 20.0}
        position_eligibility = {1: ["PG"], 2: ["SG"]}
        slot_configs = [
            {"slot_id": "PG1", "eligible_positions": ["PG"]},
            {"slot_id": "SG1", "eligible_positions": ["SG"]},
        ]

        assignment = assign_optimal_daily_lineup(
            roster_player_ids=roster,
            per_game_fp=per_game_fp,
            slot_configs=slot_configs,
            position_eligibility=position_eligibility,
            locked_assignments=None,
        )

        assert assignment["PG1"] == 1
        assert assignment["SG1"] == 2
        assert assignment.total_expected_fp == 55.0

    # --------------------------------------------------------------------------
    # M1-T24: Pre-locked slot existence validation
    # --------------------------------------------------------------------------
    def test_remediated_locked_slot_must_exist(self):
        """Pre-locked slots not in configured active slots must raise ValueError."""
        roster = [1, 2]
        positions = {1: ["PG"], 2: ["SG"]}
        fp = {1: 30.0, 2: 25.0}
        slots = {"PG": 1, "SG": 1}

        with pytest.raises(ValueError, match="does not exist"):
            assign_optimal_daily_lineup(
                roster_player_ids=roster,
                player_positions=positions,
                player_fp=fp,
                active_slots=slots,
                locked_assignments={"PHANTOM_SLOT": 1},
            )

    # --------------------------------------------------------------------------
    # M1-T25: Pre-locked player positional eligibility validation
    # --------------------------------------------------------------------------
    def test_remediated_locked_player_eligibility(self):
        """Pre-locking a player into an ineligible slot must raise ValueError."""
        roster = [1, 2]
        positions = {1: ["PG"], 2: ["C"]}
        fp = {1: 30.0, 2: 40.0}
        slots = {"PG": 1, "C": 1}

        with pytest.raises(ValueError, match="not eligible for locked slot"):
            assign_optimal_daily_lineup(
                roster_player_ids=roster,
                player_positions=positions,
                player_fp=fp,
                active_slots=slots,
                locked_assignments={"PG1": 2},
            )

    # --------------------------------------------------------------------------
    # M1-T26: Duplicate roster IDs validation
    # --------------------------------------------------------------------------
    def test_remediated_duplicate_roster_ids(self):
        """Duplicate player IDs in roster_player_ids must raise ValueError."""
        roster = [1, 1, 2]
        positions = {1: ["PG"], 2: ["SG"]}
        fp = {1: 30.0, 2: 25.0}
        slots = {"PG": 1, "SG": 1}

        with pytest.raises(ValueError, match="Duplicate player IDs in roster"):
            assign_optimal_daily_lineup(
                roster_player_ids=roster,
                player_positions=positions,
                player_fp=fp,
                active_slots=slots,
            )

    # --------------------------------------------------------------------------
    # M1-T27: Graceful None FP and None games handling
    # --------------------------------------------------------------------------
    def test_remediated_none_fp_safety(self):
        """Projections or weekly games containing None must not crash with TypeError."""
        roster = [1]
        positions = {1: ["PG"]}
        fp = {1: None}
        slots = {"PG": 1}

        daily = assign_optimal_daily_lineup(
            roster_player_ids=roster,
            player_positions=positions,
            player_fp=fp,
            active_slots=slots,
        )
        assert daily["PG1"] is None
        assert daily.total_expected_fp == 0.0

        weekly = assign_optimal_weekly_lineup(
            roster_player_ids=roster,
            player_positions=positions,
            player_fp_per_game={1: 25.0},
            player_games_in_week={1: None},
            active_slots=slots,
        )
        assert weekly["PG1"] is None
        assert weekly.total_expected_fp == 0.0

    # --------------------------------------------------------------------------
    # M1-T28: Off-day and zero-game players tiebreak
    # --------------------------------------------------------------------------
    def test_remediated_off_day_tiebreak(self):
        """Players not playing today or with 0 games must be benched rather than taking active slots."""
        roster = [1]
        positions = {1: ["PG"]}
        fp = {1: 35.0}
        slots = {"PG": 1}

        daily = assign_optimal_daily_lineup(
            roster_player_ids=roster,
            player_positions=positions,
            player_fp=fp,
            active_slots=slots,
            playing_today=set(),
        )
        assert daily["PG1"] is None
        assert daily.bench_players == [1]

        weekly = assign_optimal_weekly_lineup(
            roster_player_ids=roster,
            player_positions=positions,
            player_fp_per_game={1: 35.0},
            player_games_in_week={1: 0},
            active_slots=slots,
        )
        assert weekly["PG1"] is None
        assert weekly.bench_players == [1]


# ==============================================================================
# 4. Waiver Wire Valuation Engine Tests (F3 / R1)
# ==============================================================================

class TestWaiverValuationEngine:
    """Tests for backend.points.roster_schedule: compute_marginal_lineup_delta_fp and evaluate_waiver_candidates."""

    def test_marginal_delta_fp_basic_add_to_empty_slot(self):
        """Adding a player who plays 2 games into open active slots adds raw projected FP."""
        cal = ScheduleCalendar.from_dict({
            "BOS": ["2024-11-01", "2024-11-03"],
            "NYK": ["2024-11-02"],
            "MIA": [],
        }, player_teams={1: "BOS", 2: "NYK", 3: "MIA", 10: "BOS"})

        # Roster: 1 (PG, 30 FP), 3 (Bench, 0 games, 10 FP). Drop 3, add 10 (SG, 25 FP, 2 games).
        roster = [1, 3]
        positions = {1: ["PG"], 3: ["C"], 10: ["SG"]}
        fp = {1: 30.0, 3: 10.0, 10: 25.0}
        slots = {"PG": 1, "SG": 1}
        window = ["2024-11-01", "2024-11-02", "2024-11-03"]

        delta = compute_marginal_lineup_delta_fp(
            current_roster_ids=roster,
            drop_player_id=3,
            add_player_id=10,
            schedule=cal,
            evaluation_window=window,
            player_positions=positions,
            player_fp_per_game=fp,
            active_slots=slots,
        )
        # Player 10 starts on Nov 1 (25 FP) and Nov 3 (25 FP). Total gain = 50.0 FP.
        assert delta == 50.0

    def test_marginal_delta_fp_displaces_lower_scoring_starter(self):
        """Adding a superior player at a full position displaces the lower-scoring starter."""
        cal = ScheduleCalendar.from_dict({
            "DEN": ["2024-11-01"],
            "MIN": ["2024-11-01"],
            "MIA": [],
        }, player_teams={1: "DEN", 2: "MIA", 10: "MIN"})

        # Roster: Center 1 (20 FP, DEN plays Nov 1), Bench 2 (10 FP, MIA off).
        # Only 1 C slot. Candidate 10 (35 FP, MIN plays Nov 1).
        roster = [1, 2]
        positions = {1: ["C"], 2: ["SF"], 10: ["C"]}
        fp = {1: 20.0, 2: 10.0, 10: 35.0}
        slots = {"C": 1}
        window = ["2024-11-01"]

        delta = compute_marginal_lineup_delta_fp(
            current_roster_ids=roster,
            drop_player_id=2,
            add_player_id=10,
            schedule=cal,
            evaluation_window=window,
            player_positions=positions,
            player_fp_per_game=fp,
            active_slots=slots,
        )
        # Nov 1: baseline has Player 1 starting for 20.0 FP.
        # Modified roster has Player 10 starting for 35.0 FP (Player 1 benched).
        # Marginal Delta FP is 35.0 - 20.0 = +15.0 FP, NOT 35.0 raw points!
        assert delta == 15.0

    def test_marginal_delta_fp_cannot_displace_superior_starter(self):
        """Adding an inferior player to a full position gets benched and yields 0 Delta FP."""
        cal = ScheduleCalendar.from_dict({
            "DEN": ["2024-11-01"],
            "MIN": ["2024-11-01"],
            "MIA": [],
        }, player_teams={1: "DEN", 2: "MIA", 10: "MIN"})

        # Superstar 1 (50 FP, DEN plays Nov 1), Bench 2 (MIA off).
        # Active slot: only 1 C slot. Candidate 10 (25 FP, MIN plays Nov 1).
        roster = [1, 2]
        positions = {1: ["C"], 2: ["SF"], 10: ["C"]}
        fp = {1: 50.0, 2: 10.0, 10: 25.0}
        slots = {"C": 1}
        window = ["2024-11-01"]

        delta = compute_marginal_lineup_delta_fp(
            current_roster_ids=roster,
            drop_player_id=2,
            add_player_id=10,
            schedule=cal,
            evaluation_window=window,
            player_positions=positions,
            player_fp_per_game=fp,
            active_slots=slots,
        )
        # Candidate 10 cannot start over 50 FP starter; benched -> 0 gain
        assert delta == 0.0

    def test_marginal_delta_fp_drop_player_playing_vs_not_playing(self):
        """Dropping a player who was actively starting subtracts their lost starting points."""
        cal = ScheduleCalendar.from_dict({
            "BOS": ["2024-11-01"],
            "NYK": ["2024-11-01"],
        }, player_teams={1: "BOS", 2: "NYK"})

        # Roster: 1 (PG, 28 FP, BOS plays Nov 1).
        # Drop 1, add 2 (PG, 32 FP, NYK plays Nov 1).
        roster = [1]
        positions = {1: ["PG"], 2: ["PG"]}
        fp = {1: 28.0, 2: 32.0}
        slots = {"PG": 1}
        window = ["2024-11-01"]

        delta = compute_marginal_lineup_delta_fp(
            current_roster_ids=roster,
            drop_player_id=1,
            add_player_id=2,
            schedule=cal,
            evaluation_window=window,
            player_positions=positions,
            player_fp_per_game=fp,
            active_slots=slots,
        )
        # Net gain is 32.0 - 28.0 = +4.0 FP
        assert delta == 4.0

    def test_marginal_delta_fp_positional_flex_cascading(self):
        """Multi-position cascading allows new player to take primary slot while existing starter cascades to flex."""
        cal = ScheduleCalendar.from_dict({
            "LAL": ["2024-11-01"],
            "GSW": ["2024-11-01"],
            "BOS": ["2024-11-01"],
            "MIA": [],
        }, player_teams={1: "LAL", 2: "GSW", 3: "MIA", 10: "BOS"})

        # Roster: 1 (PG, 35 FP), 2 (SG, 30 FP), 3 (Bench, MIA off).
        # Active slots: PG: 1, SG: 1, G: 1 (flex).
        # Baseline: PG1=1 (35), SG1=2 (30). Total = 65 FP.
        # Add 10 (PG, 40 FP, BOS plays).
        # Post-add: PG1=10 (40), G1=1 (35), SG1=2 (30). Total = 105 FP.
        roster = [1, 2, 3]
        positions = {1: ["PG"], 2: ["SG"], 3: ["SF"], 10: ["PG"]}
        fp = {1: 35.0, 2: 30.0, 3: 15.0, 10: 40.0}
        slots = {"PG": 1, "SG": 1, "G": 1}
        window = ["2024-11-01"]

        delta = compute_marginal_lineup_delta_fp(
            current_roster_ids=roster,
            drop_player_id=3,
            add_player_id=10,
            schedule=cal,
            evaluation_window=window,
            player_positions=positions,
            player_fp_per_game=fp,
            active_slots=slots,
        )
        assert delta == 40.0

    def test_marginal_delta_fp_empty_window_and_identity_swap(self):
        """Empty evaluation window or swapping a player for themselves returns 0.0."""
        cal = ScheduleCalendar.from_dict({"BOS": ["2024-11-01"]}, player_teams={1: "BOS", 2: "BOS"})
        roster = [1]
        positions = {1: ["PG"], 2: ["PG"]}
        fp = {1: 30.0, 2: 30.0}
        slots = {"PG": 1}

        # Empty window
        assert compute_marginal_lineup_delta_fp(
            roster, 1, 2, cal, [], positions, fp, slots
        ) == 0.0

        # Identity swap (drop 1, add 1)
        assert compute_marginal_lineup_delta_fp(
            roster, 1, 1, cal, ["2024-11-01"], positions, fp, slots
        ) == 0.0

    def test_marginal_delta_fp_validation_errors(self):
        """Illegal drop or add candidates raise ValueError."""
        cal = ScheduleCalendar.from_dict({"BOS": ["2024-11-01"]}, player_teams={1: "BOS", 2: "BOS", 3: "BOS"})
        roster = [1, 2]
        positions = {1: ["PG"], 2: ["SG"], 3: ["SF"]}
        fp = {1: 30.0, 2: 25.0, 3: 20.0}
        slots = {"PG": 1, "SG": 1}

        # Drop player not on roster
        with pytest.raises(ValueError, match="not in current roster"):
            compute_marginal_lineup_delta_fp(
                roster, drop_player_id=999, add_player_id=3,
                schedule=cal, evaluation_window=["2024-11-01"],
                player_positions=positions, player_fp_per_game=fp, active_slots=slots,
            )

        # Add player already on roster
        with pytest.raises(ValueError, match="already in current roster"):
            compute_marginal_lineup_delta_fp(
                roster, drop_player_id=1, add_player_id=2,
                schedule=cal, evaluation_window=["2024-11-01"],
                player_positions=positions, player_fp_per_game=fp, active_slots=slots,
            )

    def test_marginal_lineup_detailed_breakdown(self):
        """Detailed breakdown returns daily baseline, daily modified, and games played."""
        cal = ScheduleCalendar.from_dict({
            "BOS": ["2024-11-01", "2024-11-02"],
            "NYK": ["2024-11-02"],
            "MIA": [],
        }, player_teams={1: "BOS", 2: "MIA", 10: "NYK"})

        roster = [1, 2]
        positions = {1: ["PG"], 2: ["C"], 10: ["C"]}
        fp = {1: 30.0, 2: 15.0, 10: 22.0}
        slots = {"PG": 1, "C": 1}
        window = ["2024-11-01", "2024-11-02"]

        detailed = compute_marginal_lineup_detailed(
            current_roster_ids=roster,
            drop_player_id=2,
            add_player_id=10,
            schedule=cal,
            evaluation_window=window,
            player_positions=positions,
            player_fp_per_game=fp,
            active_slots=slots,
        )
        assert detailed["candidate_games"] == 1
        assert detailed["dropped_games"] == 0
        assert detailed["daily_baseline"]["2024-11-01"] == 30.0
        assert detailed["daily_modified"]["2024-11-02"] == 52.0  # 30 + 22
        assert detailed["delta_fp"] == 22.0

    def test_evaluate_waiver_candidates_batch_ranking(self):
        """evaluate_waiver_candidates ranks candidates by Delta FP descending."""
        cal = ScheduleCalendar.from_dict({
            "BOS": ["2024-11-01", "2024-11-02", "2024-11-03"],  # 3 games
            "NYK": ["2024-11-01", "2024-11-03"],                # 2 games
            "LAL": ["2024-11-02"],                              # 1 game
            "MIA": [],
        }, player_teams={1: "MIA", 101: "BOS", 102: "NYK", 103: "LAL"})

        roster = [1]  # Bench player with 0 games
        candidates = [101, 102, 103]
        positions = {1: ["Util"], 101: ["Util"], 102: ["Util"], 103: ["Util"]}
        fp = {1: 10.0, 101: 25.0, 102: 30.0, 103: 20.0}
        slots = {"Util": 1}
        window = ["2024-11-01", "2024-11-02", "2024-11-03"]

        results = evaluate_waiver_candidates(
            current_roster_ids=roster,
            candidate_ids=candidates,
            drop_player_id=1,
            schedule=cal,
            evaluation_window=window,
            player_positions=positions,
            player_fp_per_game=fp,
            active_slots=slots,
        )
        # Candidate 101: 3 * 25 = 75 FP
        # Candidate 102: 2 * 30 = 60 FP
        # Candidate 103: 1 * 20 = 20 FP
        assert len(results) == 3
        assert results[0].candidate_id == 101
        assert results[0].delta_fp == 75.0
        assert results[0].candidate_games == 3
        assert results[1].candidate_id == 102
        assert results[1].delta_fp == 60.0
        assert results[2].candidate_id == 103
        assert results[2].delta_fp == 20.0

    def test_evaluate_waiver_candidates_multiple_drop_candidates(self):
        """When multiple drop candidates exist, evaluator finds the optimal drop player per candidate."""
        cal = ScheduleCalendar.from_dict({
            "BOS": ["2024-11-01"],
            "MIA": [],
            "GSW": ["2024-11-01"],
        }, player_teams={1: "BOS", 2: "MIA", 101: "GSW"})

        # Player 1 is starter (30 FP), Player 2 is bench (0 FP, off)
        roster = [1, 2]
        positions = {1: ["Util"], 2: ["Util"], 101: ["Util"]}
        fp = {1: 30.0, 2: 10.0, 101: 35.0}
        slots = {"Util": 1}
        window = ["2024-11-01"]

        # If dropping 1: delta = 35 - 30 = +5.0
        # If dropping 2: delta = 35 - 30 = +5.0 (since only 1 Util slot, 101 starts, 1 is benched)
        results = evaluate_waiver_candidates(
            current_roster_ids=roster,
            candidate_ids=[101],
            drop_candidates=[1, 2],
            schedule=cal,
            evaluation_window=window,
            player_positions=positions,
            player_fp_per_game=fp,
            active_slots=slots,
        )
        assert len(results) == 1
        assert results[0].delta_fp == 5.0
        # Dict compatibility test
        assert results[0]["delta_fp"] == 5.0
        assert "candidate_id" in results[0].to_dict()


# ==============================================================================
# 5. Streamer Optimization Engine Tests (F4 / Follow-up R1, R4)
# ==============================================================================

class TestStreamerOptimizationEngine:
    """Tests for backend.points.streamer: optimize_streaming_acquisitions and schedule density analysis."""

    def test_density_analysis_light_slate_and_congestion(self):
        """Schedule density analysis accurately identifies light slates and starter slot congestion."""
        cal = ScheduleCalendar.from_dict({
            "BOS": ["2024-11-01"],
            "NYK": ["2024-11-01"],
            "LAL": ["2024-11-02"],
            "GSW": ["2024-11-02"],
            "MIA": ["2024-11-02"],
            "CHI": ["2024-11-02"],
        }, player_teams={1: "BOS", 2: "LAL", 3: "GSW"})

        roster = [1, 2, 3]
        positions = {1: ["PG"], 2: ["SG"], 3: ["SF"]}
        fp = {1: 30.0, 2: 30.0, 3: 30.0}
        slots = {"PG": 1, "SG": 1, "SF": 1}
        window = ["2024-11-01", "2024-11-02"]

        density = analyze_schedule_density(
            current_roster_ids=roster,
            schedule=cal,
            evaluation_window=window,
            player_positions=positions,
            player_fp_per_game=fp,
            active_slots=slots,
        )
        # Nov 1: only Player 1 playing -> 2 open slots -> light slate
        assert density["2024-11-01"]["open_slots_count"] == 2
        assert density["2024-11-01"]["is_light_slate"] is True

        # Nov 2: Player 2 and 3 playing -> 1 open slot -> light slate
        assert density["2024-11-02"]["open_slots_count"] == 1

    def test_streamer_daily_single_slot_churn(self):
        """Daily single-slot churn rotates through daily adds under acquisition limit."""
        cal = ScheduleCalendar.from_dict({
            "ATL": ["2024-11-01"],
            "BOS": ["2024-11-02"],
            "CLE": ["2024-11-03"],
            "MIA": [],
        }, player_teams={1: "MIA", 101: "ATL", 102: "BOS", 103: "CLE"})

        roster = [1]  # Dropped bench player
        candidates = [101, 102, 103]
        positions = {1: ["Util"], 101: ["Util"], 102: ["Util"], 103: ["Util"]}
        fp = {1: 10.0, 101: 25.0, 102: 25.0, 103: 25.0}
        slots = {"Util": 1}
        window = ["2024-11-01", "2024-11-02", "2024-11-03"]

        res = optimize_streaming_acquisitions(
            current_roster_ids=roster,
            available_waiver_ids=candidates,
            schedule=cal,
            evaluation_window=window,
            player_positions=positions,
            player_fp_per_game=fp,
            active_slots=slots,
            max_acquisitions=3,
            mode="daily_churn",
        )
        plan = res.optimal_plan
        assert plan.strategy == "daily_churn"
        assert plan.total_acquisitions == 3
        assert plan.total_games_streamed == 3
        assert plan.total_projected_fp == 75.0
        assert plan.points_per_add == 25.0

    def test_streamer_tactical_multi_back_to_back(self):
        """Tactical multi mode targets back-to-backs (2 games in 2 days) to maximize points per add."""
        cal = ScheduleCalendar.from_dict({
            "DEN": ["2024-11-01", "2024-11-02"],  # Back-to-back
            "MIA": [],
        }, player_teams={1: "MIA", 101: "DEN"})

        roster = [1]
        positions = {1: ["Util"], 101: ["Util"]}
        fp = {1: 10.0, 101: 28.0}
        slots = {"Util": 1}
        window = ["2024-11-01", "2024-11-02"]

        res = optimize_streaming_acquisitions(
            current_roster_ids=roster,
            available_waiver_ids=[101],
            schedule=cal,
            evaluation_window=window,
            player_positions=positions,
            player_fp_per_game=fp,
            active_slots=slots,
            max_acquisitions=1,
            mode="tactical_multi",
        )
        plan = res.optimal_plan
        assert plan.strategy == "tactical_multi"
        assert plan.total_acquisitions == 1
        assert plan.total_games_streamed == 2
        assert plan.total_projected_fp == 56.0
        assert plan.points_per_add == 56.0  # 2 games for 1 add = double PPA

    def test_streamer_tactical_multi_3_in_4_nights(self):
        """Tactical multi mode holds player for 3 games in 4 nights on a single acquisition."""
        cal = ScheduleCalendar.from_dict({
            "PHX": ["2024-11-01", "2024-11-03", "2024-11-04"],  # 3 in 4
            "MIA": [],
        }, player_teams={1: "MIA", 101: "PHX"})

        roster = [1]
        positions = {1: ["Util"], 101: ["Util"]}
        fp = {1: 10.0, 101: 22.0}
        slots = {"Util": 1}
        window = ["2024-11-01", "2024-11-02", "2024-11-03", "2024-11-04"]

        res = optimize_streaming_acquisitions(
            current_roster_ids=roster,
            available_waiver_ids=[101],
            schedule=cal,
            evaluation_window=window,
            player_positions=positions,
            player_fp_per_game=fp,
            active_slots=slots,
            max_acquisitions=1,
            mode="tactical_multi",
        )
        plan = res.optimal_plan
        assert plan.total_acquisitions == 1
        assert plan.total_games_streamed == 3
        assert plan.total_projected_fp == 66.0
        assert plan.points_per_add == 66.0

    def test_streamer_multi_slot_streaming(self):
        """Multi-slot streaming allows concurrent streaming slots across the week."""
        cal = ScheduleCalendar.from_dict({
            "BOS": ["2024-11-01", "2024-11-03"],  # Slot 1
            "NYK": ["2024-11-02", "2024-11-04"],  # Slot 2
            "MIA": [],
        }, player_teams={1: "MIA", 2: "MIA", 101: "BOS", 102: "NYK"})

        roster = [1, 2]  # Two drop candidates
        positions = {1: ["Util"], 2: ["Util"], 101: ["Util"], 102: ["Util"]}
        fp = {1: 10.0, 2: 10.0, 101: 25.0, 102: 25.0}
        slots = {"Util": 2}
        window = ["2024-11-01", "2024-11-02", "2024-11-03", "2024-11-04"]

        res = optimize_streaming_acquisitions(
            current_roster_ids=roster,
            available_waiver_ids=[101, 102],
            schedule=cal,
            evaluation_window=window,
            player_positions=positions,
            player_fp_per_game=fp,
            active_slots=slots,
            max_acquisitions=2,
            streaming_slots=2,
            mode="tactical_multi",
        )
        plan = res.optimal_plan
        assert plan.total_acquisitions == 2
        assert plan.total_games_streamed == 4
        assert plan.total_projected_fp == 100.0

    def test_streamer_acquisition_limit_strict_enforcement(self):
        """Weekly acquisition cap is strictly enforced even when more profitable days exist."""
        cal = ScheduleCalendar.from_dict({
            "ATL": ["2024-11-01"],
            "BOS": ["2024-11-02"],
            "CLE": ["2024-11-03"],
            "DAL": ["2024-11-04"],
            "DEN": ["2024-11-05"],
            "MIA": [],
        }, player_teams={1: "MIA", 101: "ATL", 102: "BOS", 103: "CLE", 104: "DAL", 105: "DEN"})

        roster = [1]
        candidates = [101, 102, 103, 104, 105]
        positions = {i: ["Util"] for i in [1, 101, 102, 103, 104, 105]}
        # FP per game: 105 is best (40 FP), 104 is 35, 103 is 30, 102 is 25, 101 is 20
        fp = {1: 10.0, 101: 20.0, 102: 25.0, 103: 30.0, 104: 35.0, 105: 40.0}
        slots = {"Util": 1}
        window = ["2024-11-01", "2024-11-02", "2024-11-03", "2024-11-04", "2024-11-05"]

        res = optimize_streaming_acquisitions(
            current_roster_ids=roster,
            available_waiver_ids=candidates,
            schedule=cal,
            evaluation_window=window,
            player_positions=positions,
            player_fp_per_game=fp,
            active_slots=slots,
            max_acquisitions=2,
            mode="daily_churn",
        )
        plan = res.optimal_plan
        assert plan.total_acquisitions == 2
        # Should pick the 2 highest yield adds (40.0 and 35.0 = 75.0 FP)
        assert plan.total_projected_fp == 75.0
        assert {m.add_player_id for m in plan.moves} == {104, 105}

    def test_streamer_mode_auto_prefers_tactical_efficiency(self):
        """Mode auto selects tactical multi when it yields equal or superior points with fewer adds."""
        cal = ScheduleCalendar.from_dict({
            "DEN": ["2024-11-01", "2024-11-02"],  # B2B
            "MIA": [],
        }, player_teams={1: "MIA", 101: "DEN"})

        roster = [1]
        positions = {1: ["Util"], 101: ["Util"]}
        fp = {1: 10.0, 101: 25.0}
        slots = {"Util": 1}
        window = ["2024-11-01", "2024-11-02"]

        res = optimize_streaming_acquisitions(
            current_roster_ids=roster,
            available_waiver_ids=[101],
            schedule=cal,
            evaluation_window=window,
            player_positions=positions,
            player_fp_per_game=fp,
            active_slots=slots,
            max_acquisitions=7,
            mode="auto",
        )
        assert res.optimal_plan.strategy == "tactical_multi"
        assert res.points_per_add == 50.0
        assert "daily_churn" in res.alternative_plans
        assert "tactical_multi" in res.alternative_plans

    def test_streamer_crowded_slate_avoidance(self):
        """Streamers are benched and not streamed on slates where all slots are filled by better starters."""
        cal = ScheduleCalendar.from_dict({
            "LAL": ["2024-11-01"],
            "DEN": ["2024-11-01"],
            "BOS": ["2024-11-02"],
            "MIA": [],
        }, player_teams={1: "LAL", 2: "MIA", 101: "DEN", 102: "BOS"})

        # On Nov 1: Roster has starter 1 (45 FP). Slot: 1 C.
        # Candidate 101 (20 FP) plays Nov 1, Candidate 102 (20 FP) plays Nov 2 (when starter 1 is off).
        roster = [1, 2]
        positions = {1: ["C"], 2: ["SF"], 101: ["C"], 102: ["C"]}
        fp = {1: 45.0, 2: 10.0, 101: 20.0, 102: 20.0}
        slots = {"C": 1}
        window = ["2024-11-01", "2024-11-02"]

        res = optimize_streaming_acquisitions(
            current_roster_ids=roster,
            available_waiver_ids=[101, 102],
            schedule=cal,
            evaluation_window=window,
            player_positions=positions,
            player_fp_per_game=fp,
            active_slots=slots,
            max_acquisitions=2,
            mode="daily_churn",
        )
        # Nov 1 is crowded (candidate 101 gets benched behind 45 FP starter).
        # Nov 2 has an open C slot, so candidate 102 starts and scores 20 FP!
        assert res.optimal_plan.total_acquisitions == 1
        assert res.optimal_plan.moves[0].date == "2024-11-02"
        assert res.optimal_plan.moves[0].add_player_id == 102

    def test_streamer_boundary_cases(self):
        """Streamer handles empty windows, zero max acquisitions, and empty waivers safely."""
        cal = ScheduleCalendar.from_dict({"BOS": ["2024-11-01"]}, player_teams={1: "BOS"})
        roster = [1]
        positions = {1: ["PG"]}
        fp = {1: 30.0}
        slots = {"PG": 1}

        # 0 acquisitions
        res_zero = optimize_streaming_acquisitions(
            roster, [2], cal, ["2024-11-01"], positions, fp, slots, max_acquisitions=0
        )
        assert res_zero.total_acquisitions_used == 0

        # Empty waivers
        res_empty = optimize_streaming_acquisitions(
            roster, [], cal, ["2024-11-01"], positions, fp, slots, max_acquisitions=5
        )
        assert res_empty.total_acquisitions_used == 0

        # Empty window
        res_no_window = optimize_streaming_acquisitions(
            roster, [2], cal, [], positions, fp, slots, max_acquisitions=5
        )
        assert res_no_window.total_acquisitions_used == 0

    def test_streamer_dict_compatibility(self):
        """Result and Plan objects support mapping operations and serialization."""
        cal = ScheduleCalendar.from_dict({"BOS": ["2024-11-01"]}, player_teams={1: "BOS", 2: "BOS"})
        res = optimize_streaming_acquisitions(
            [1], [2], cal, ["2024-11-01"], {1: ["PG"], 2: ["PG"]}, {1: 10.0, 2: 25.0}, {"PG": 1}
        )
        d = res.to_dict()
        assert "optimal_plan" in d
        assert "total_acquisitions_used" in d
        assert res["total_acquisitions_used"] == d["total_acquisitions_used"]
        assert len(res) > 0


# ==============================================================================
# 6. Waiver & Streamer Edge Cases & Real-World Scenarios (Tiers 2, 3, 4)
# ==============================================================================

class TestWaiverStreamerIntegrationAndEdgeCases:
    """Complex scenarios, synthetic schedules, and multi-tier edge cases."""

    def test_real_world_matchup_streamer_churn_vs_hold(self):
        """Simulate full 7-day matchup week testing realistic streaming volume."""
        cal = ScheduleCalendar()
        # Pick 3 active teams from synthetic calendar
        teams = ["BOS", "LAL", "GSW"]
        window = cal.normalize_window(("2024-10-21", "2024-10-27"))

        # User team has core stars
        cal.register_player_team(1, "BOS")
        cal.register_player_team(2, "LAL")
        cal.register_player_team(3, "GSW")
        cal.register_player_team(4, "MIA")  # Droppable bench player

        # Streamers available
        cal.register_player_team(101, "DEN")
        cal.register_player_team(102, "NYK")
        cal.register_player_team(103, "PHX")

        roster = [1, 2, 3, 4]
        positions = {1: ["PG"], 2: ["SG"], 3: ["SF"], 4: ["Util"], 101: ["Util"], 102: ["Util"], 103: ["Util"]}
        fp = {1: 40.0, 2: 38.0, 3: 35.0, 4: 15.0, 101: 24.0, 102: 25.0, 103: 26.0}
        slots = {"PG": 1, "SG": 1, "SF": 1, "Util": 1}

        res = optimize_streaming_acquisitions(
            current_roster_ids=roster,
            available_waiver_ids=[101, 102, 103],
            schedule=cal,
            evaluation_window=window,
            player_positions=positions,
            player_fp_per_game=fp,
            active_slots=slots,
            max_acquisitions=7,
            drop_candidate_ids=[4],
            mode="auto",
        )
        assert res.total_acquisitions_used > 0
        assert res.total_streamed_games > 0
        assert res.total_net_delta_fp > 0.0

    def test_waiver_candidates_all_negative_or_zero_fp(self):
        """Waiver candidates with <= 0 FP are excluded from streaming."""
        cal = ScheduleCalendar.from_dict({"BOS": ["2024-11-01"]}, player_teams={1: "BOS", 101: "BOS"})
        res = optimize_streaming_acquisitions(
            [1], [101], cal, ["2024-11-01"], {1: ["PG"], 101: ["PG"]}, {1: 20.0, 101: 0.0}, {"PG": 1}
        )
        assert res.total_acquisitions_used == 0

    def test_displaced_starter_cascading_with_ir_and_bench(self):
        """Marginal delta correctly handles rosters with injured IR players."""
        cal = ScheduleCalendar.from_dict({
            "BOS": ["2024-11-01"],
            "NYK": ["2024-11-01"],
        }, player_teams={1: "BOS", 2: "NYK", 3: "BOS"})

        # Player 2 is injured (in IR). Player 1 is starter. Drop player 3 (bench). Add player 10 (starter).
        cal.register_player_team(10, "BOS")
        roster = [1, 2, 3]
        positions = {1: ["PG"], 2: ["SG"], 3: ["SF"], 10: ["SG"]}
        fp = {1: 30.0, 2: 0.0, 3: 10.0, 10: 28.0}
        slots = {"PG": 1, "SG": 1}

        delta = compute_marginal_lineup_delta_fp(
            current_roster_ids=roster,
            drop_player_id=3,
            add_player_id=10,
            schedule=cal,
            evaluation_window=["2024-11-01"],
            player_positions=positions,
            player_fp_per_game=fp,
            active_slots=slots,
            injured_players={2},
        )
        # Nov 1: baseline has Player 1 in PG1 (30 FP), SG1 empty (Player 2 injured). Total = 30.
        # Modified has Player 1 in PG1 (30 FP) and Player 10 in SG1 (28 FP). Total = 58.
        # Delta = 28.0.
        assert delta == 28.0

    def test_marginal_delta_fp_none_and_nan_in_waiver_candidate(self):
        """Waiver candidate with None or NaN projected FP is handled without TypeError."""
        cal = ScheduleCalendar.from_dict({"BOS": ["2024-11-01"]}, player_teams={1: "BOS", 10: "BOS"})
        roster = [1]
        positions = {1: ["PG"], 10: ["PG"]}
        fp = {1: 30.0, 10: None}
        slots = {"PG": 1}

        delta = compute_marginal_lineup_delta_fp(
            current_roster_ids=roster,
            drop_player_id=1,
            add_player_id=10,
            schedule=cal,
            evaluation_window=["2024-11-01"],
            player_positions=positions,
            player_fp_per_game=fp,
            active_slots=slots,
        )
        # Dropping 30 FP starter for None FP player yields -30.0 Delta FP
        assert delta == -30.0

    def test_marginal_delta_fp_multi_week_window(self):
        """Multi-week 14-day evaluation window accurately accumulates daily starting FP deltas."""
        cal = ScheduleCalendar()
        cal.register_player_team(1, "BOS")
        cal.register_player_team(2, "MIA")
        cal.register_player_team(10, "LAL")

        roster = [1, 2]
        positions = {1: ["PG"], 2: ["Util"], 10: ["Util"]}
        fp = {1: 40.0, 2: 10.0, 10: 25.0}
        slots = {"PG": 1, "Util": 1}
        window = cal.normalize_window(("2024-10-21", "2024-11-03"))  # 14 days

        delta = compute_marginal_lineup_delta_fp(
            current_roster_ids=roster,
            drop_player_id=2,
            add_player_id=10,
            schedule=cal,
            evaluation_window=window,
            player_positions=positions,
            player_fp_per_game=fp,
            active_slots=slots,
        )
        assert delta > 0.0

    def test_evaluate_waiver_candidates_top_n_truncation(self):
        """top_n parameter limits the number of returned waiver candidates."""
        cal = ScheduleCalendar.from_dict({
            "BOS": ["2024-11-01"],
            "NYK": ["2024-11-01"],
            "LAL": ["2024-11-01"],
            "MIA": [],
        }, player_teams={1: "MIA", 101: "BOS", 102: "NYK", 103: "LAL"})

        roster = [1]
        candidates = [101, 102, 103]
        positions = {i: ["Util"] for i in [1, 101, 102, 103]}
        fp = {1: 10.0, 101: 30.0, 102: 25.0, 103: 20.0}
        slots = {"Util": 1}

        res = evaluate_waiver_candidates(
            current_roster_ids=roster,
            candidate_ids=candidates,
            drop_player_id=1,
            schedule=cal,
            evaluation_window=["2024-11-01"],
            player_positions=positions,
            player_fp_per_game=fp,
            active_slots=slots,
            top_n=2,
        )
        assert len(res) == 2
        assert res[0].candidate_id == 101
        assert res[1].candidate_id == 102

    def test_streamer_all_players_ineligible_for_open_slots(self):
        """When candidates cannot play the required open positions, zero acquisitions are made."""
        cal = ScheduleCalendar.from_dict({"BOS": ["2024-11-01"]}, player_teams={1: "BOS", 101: "BOS"})
        # Only open slot is Center. Candidate is Point Guard with no C or Util eligibility.
        roster = [1]
        positions = {1: ["C"], 101: ["PG"]}
        fp = {1: 10.0, 101: 30.0}
        slots = {"C": 1}

        res = optimize_streaming_acquisitions(
            current_roster_ids=roster,
            available_waiver_ids=[101],
            schedule=cal,
            evaluation_window=["2024-11-01"],
            player_positions=positions,
            player_fp_per_game=fp,
            active_slots=slots,
            max_acquisitions=5,
        )
        assert res.total_acquisitions_used == 0

    def test_streamer_drop_candidate_priority(self):
        """When multiple drop candidates exist, optimizer selects the one whose loss is lowest."""
        cal = ScheduleCalendar.from_dict({
            "BOS": ["2024-11-01"],
            "NYK": ["2024-11-01"],
            "DEN": ["2024-11-02"],
        }, player_teams={1: "BOS", 2: "NYK", 101: "DEN"})

        # Player 1 is starter (35 FP on Nov 1). Player 2 is starter (30 FP on Nov 1).
        # Dropping Player 2 costs 30 FP, dropping Player 1 costs 35 FP.
        # Streamer 101 plays Nov 2.
        roster = [1, 2]
        positions = {1: ["PG"], 2: ["SG"], 101: ["Util"]}
        fp = {1: 35.0, 2: 30.0, 101: 25.0}
        slots = {"PG": 1, "SG": 1, "Util": 1}
        window = ["2024-11-01", "2024-11-02"]

        res = optimize_streaming_acquisitions(
            current_roster_ids=roster,
            available_waiver_ids=[101],
            schedule=cal,
            evaluation_window=window,
            player_positions=positions,
            player_fp_per_game=fp,
            active_slots=slots,
            max_acquisitions=1,
            drop_candidate_ids=[1, 2],
        )
        # Designated drop should be 2 (the lower scoring starter)
        assert res.optimal_plan.moves[0].drop_player_id == 2


