// api/client.ts
// HTTP client for the Fantasy Basketball Optimizer backend.
// All fetch calls go through this module; callers receive typed results.

import { PlayerResult, PlayerGScore, SessionRequest, SportConfig, LeagueTradeImpact, StreamerOptimizeResponse } from '../types.js'
import { setPlayerRegistry } from '../player_registry.js'
import { setScoringSystem } from '../app_state.js'


// Empty string = same-origin deployment (frontend and backend served from the same host).
// Change to an absolute URL (e.g. 'https://api.example.com') if the frontend
// and backend are ever deployed on different origins.
export const BASE_URL = ''


// ── HTTP helpers ──────────────────────────────────────────────────────────────

/** Thrown by jsonRequest when the backend returns a non-2xx status. Callers that
 *  want to react to specific statuses (e.g. retry on session expiry) should catch
 *  this and check `.status` rather than parsing the message string. */
export class HTTPError extends Error {
    constructor(
        public readonly status: number
        , public readonly body: string
        , label: string
    ) {
        super(`${label} failed (${status}): ${body}`)
        this.name = 'HTTPError'
    }
}

/** FastAPI wraps an error message as {"detail": "..."}. Returns that sentence, or the body
 *  unchanged when it is not shaped that way — a caller showing this to a person wants the
 *  message, not the envelope. */
function readErrorDetail(body: string): string {
    try {
        const parsed = JSON.parse(body)
        return typeof parsed?.detail === 'string' ? parsed.detail : body
    } catch {
        return body
    }
}

// In-flight backend work, visible to background consumers: the headshot preloader
// yields to bursty session traffic (evaluate batches and their follow-ups) and sweeps
// only while this is zero. Session BUILDS — the bare POST /sessions and the PATCH
// /sessions/{id} rebuild — are deliberately not counted: each is one long CPU-bound
// call on the server while image serving is pure I/O, so the build window is free
// time the sweep should use.
let inFlightSessionRequests = 0

/** Number of counted session requests in flight (0 = the backend is quiet apart from,
 *  at most, a session build/rebuild). */
export function countInFlightSessionRequests(): number {
    return inFlightSessionRequests
}

/** Issues `fetch(url, init)` and parses the JSON response, throwing HTTPError on
 *  non-2xx. The `label` prefixes error messages so callers don't each repeat the
 *  "X failed (status): body" boilerplate. */
async function jsonRequest<T>(
    url: string
    , label: string
    , init?: RequestInit
): Promise<T> {
    const isSessionRequest = url.includes('/sessions/')
        && (init?.method ?? 'GET').toUpperCase() !== 'PATCH'
    if (isSessionRequest) inFlightSessionRequests += 1
    try {
        const res = await fetch(url, init)
        if (!res.ok) {
            const body = await res.text()
            // A rate-limited response already carries a sentence written for a person ("try
            // again in N seconds"). Surfacing that instead of the raw JSON body is the
            // difference between an explanation and an opaque failure.
            if (res.status === 429) throw new HTTPError(res.status, readErrorDetail(body), label)
            throw new HTTPError(res.status, body, label)
        }
        if (res.status === 204) return undefined as T   // no content (e.g. token exchange)
        return await (res.json() as Promise<T>)
    } finally {
        if (isSessionRequest) inFlightSessionRequests -= 1
    }
}


// ── Map backend Candidate → frontend PlayerResult ─────────────────────────────


// Adapter between backend snake_case Candidate objects and frontend PlayerResult objects.
// The backend follows Python naming conventions; this is the single place where that translation
// happens. Players are keyed by player id end to end; display resolution happens in the
// rendering layer via the session registry.
/** Converts raw backend Candidate objects to frontend PlayerResult objects, remapping snake_case keys to camelCase. */
export function candidatesToPlayerResults(candidates: any[]): PlayerResult[] {
    return candidates.filter(c =>
        c.h_score != null ||
        c.points_per_game != null ||
        c.projected_fp_per_game != null ||
        c.marginal_fp != null ||
        c.delta_fp != null
    ).map((c, i) => ({
        player_id:        c.player_id,
        h_score:          c.h_score ?? 0,
        h_rank:           c.h_rank ?? (c.rank ?? i + 1),
        g_rank:           c.g_rank ?? (c.rank ?? i + 1),
        win_rates:        c.win_rates ?? [],
        category_weights: c.category_weights ?? [],
        g_score_rows: c.g_score_rows ? c.g_score_rows.map((r: any) => ({
            label:   r.label,
            values:  r.values,
            total:   r.total,
            isTotal: r.is_total,
        })) : [],
        flex_allocations: c.flex_allocations ? {
            base_positions: c.flex_allocations.base_positions,
            rows: c.flex_allocations.rows.map((r: any) => ({
                label:   r.label,
                values:  r.values,
                isTotal: r.is_total,
            })),
        } : undefined,
        roster: c.roster ? {
            slots: c.roster.slots,
            assignments: Object.fromEntries(
                Object.entries(c.roster.assignments).map(([slot, a]: [string, any]) => [
                    slot,
                    a ? {
                        player_id:   a.player_id,
                        isCandidate: a.is_candidate,
                    } : null,
                ]),
            ),
        } : undefined,
        auction_values: c.auction_values ? {
            your_dollar:   c.auction_values.your_dollar,
            gnrc_dollar:   c.auction_values.gnrc_dollar,
            orig_dollar:   c.auction_values.orig_dollar,
            gnrc_dollar_g: c.auction_values.gnrc_dollar_g,
            orig_dollar_g: c.auction_values.orig_dollar_g,
        } : undefined,
        // Points mode fields
        points_per_game:        c.points_per_game ?? c.projected_fp_per_game,
        total_projected_points: c.total_projected_points ?? c.projected_fp_total,
        marginal_fp:            c.marginal_fp,
        delta_fp:               c.delta_fp ?? c.marginal_fp,
        candidate_games:        c.candidate_games,
        dropped_games:          c.dropped_games,
        daily_deltas:           c.daily_deltas,
    }))
}

// ── GET /config/{sport} ──────────────────────────────────────────────────────

/** Fetches sport-specific configuration (defaults, categories, positions) from parameters.yaml. */
export async function fetchConfig(sport: string): Promise<SportConfig> {
    return jsonRequest(`${BASE_URL}/config/${encodeURIComponent(sport)}`, 'Config fetch')
}

// ── POST /data/upload ─────────────────────────────────────────────────────────

/** Uploads a projection CSV (headers interpreted server-side) and returns its data_id.
 *  missing_stats lists standard stat columns the file does not carry — a source that pairs
 *  projections with a league only includes that league's active categories. */
export async function uploadCsv(
    file: File
): Promise<{ data_id: string; n_players: number; expires_at: string; missing_stats: string[] }> {
    const form = new FormData()
    form.append('file', file)
    return jsonRequest(`${BASE_URL}/data/upload`, 'Upload', { method: 'POST', body: form })
}

// ── GET /seasons ───────────────────────────────────────────────────────────────

/** Fetches the list of available historical seasons from the backend. */
export async function getSeasons(): Promise<string[]> {
    const data = await jsonRequest<{ seasons: string[] }>(`${BASE_URL}/seasons`, 'Get seasons')
    return data.seasons
}

// ── GET /players/pool-ids ─────────────────────────────────────────────────────

/** Best-effort NBA id list for a data source's pool, for prefetching headshots while
 *  the session builds. The backend answers only from already-cached frames — an empty
 *  list means "not cheaply known yet" and the post-build registry sweep covers it. */
export async function fetchPoolPlayerIds(
    dataType: string
    , season?: string | null
): Promise<number[]> {
    const seasonParam = season ? `&season=${encodeURIComponent(season)}` : ''
    const data = await jsonRequest<{ player_ids: number[] }>(
        `${BASE_URL}/players/pool-ids?data_type=${encodeURIComponent(dataType)}${seasonParam}`,
        'Pool ids fetch',
    )
    return data.player_ids
}

// ── POST /sessions ─────────────────────────────────────────────────────────────

/** Installs the payload's registry, then returns its id-keyed G-score rows. */
function adoptRegistryAndGScores(body: any): PlayerGScore[] {
    setPlayerRegistry(body.players)
    return body.g_scores.map((entry: any) => ({
        player_id: entry.player_id,
        total:     entry.total,
        values:    entry.values,
    }))
}

/** Creates a new backend session, running the full 5-step pipeline. Returns session_id and resolved categories. */
export async function createSession(
    req: SessionRequest
    , signal?: AbortSignal
): Promise<{ session_id: string; categories: string[]; g_scores: PlayerGScore[]; n_players_loaded: number; expires_at: string; scoring_system?: 'categories' | 'points' }> {
    const body: any = await jsonRequest(`${BASE_URL}/sessions`, 'Create session', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(req),
        signal,
    })
    if (body.scoring_system) {
        setScoringSystem(body.scoring_system)
    }
    return { ...body, g_scores: adoptRegistryAndGScores(body) }
}

// ── PATCH /sessions/{id} ──────────────────────────────────────────────────────

/** Patches an existing session. Only call via `createOrPatchSession`, which handles 404 (expired session) recovery. */
export async function patchSession(
    sessionId: string
    , req: Record<string, unknown>
    , signal?: AbortSignal
): Promise<{ ok: boolean; steps_rerun: number[] }> {
    return jsonRequest(`${BASE_URL}/sessions/${sessionId}`, 'Patch session', {
        method:  'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(req),
        signal,
    })
}

// ── GET /sessions/{id}/g-scores ───────────────────────────────────────────────

/** Fetches the current G-scores for a session directly from session state. */
export async function fetchGScores(sessionId: string): Promise<PlayerGScore[]> {
    const data: any = await jsonRequest(`${BASE_URL}/sessions/${sessionId}/g-scores`, 'Fetch G-scores')
    return adoptRegistryAndGScores(data)
}

// ── POST /sessions/{id}/evaluate ──────────────────────────────────────────────

/** Runs the H-score algorithm for the given draft/auction state and returns ranked candidates. Supports AbortSignal for cancellation. */
export async function evaluate(
    sessionId: string
    , req: {
        player_assignments: Record<string, number[]>
        my_team_id: string
        exclusion_list?: number[]
        remaining_cash?: Record<string, number>
        candidate_offset?: number   // draft/waiver batching: slice start
        candidate_limit?: number    // draft/waiver batching: slice size (omit = whole pool)
    }
    , signal?: AbortSignal
): Promise<{ iteration: number; candidates: any[]; has_more?: boolean; total_candidates?: number }> {
    return jsonRequest(`${BASE_URL}/sessions/${sessionId}/evaluate`, 'Evaluate', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(req),
        signal,
    })
}

// ── POST /sessions/{id}/trade/analyze ────────────────────────────────────────

export interface TeamHScore {
    h_score: number
    rates: number[]
}

export interface TeamTradeResult {
    pre: TeamHScore
    post: TeamHScore
}

export interface TradeAnalyzeResponse {
    your_team?: TeamTradeResult | null
    their_team?: TeamTradeResult | null
    error?: string | null
    // Points mode fields
    scoring_system?: 'categories' | 'points'
    team_a_pre_fp?: number
    team_a_post_fp?: number
    team_a_delta_fp?: number
    team_a_net_delta_fp?: number
    team_a_streamer_boost?: number
    team_b_pre_fp?: number
    team_b_post_fp?: number
    team_b_delta_fp?: number
    team_b_net_delta_fp?: number
    team_b_streamer_boost?: number
    is_legal?: boolean
    rejection_reason?: string | null
    team_a_opened_spots?: number
    team_b_opened_spots?: number
    league_impact?: LeagueTradeImpact | null
    your_net_delta_fp?: number | null
    their_net_delta_fp?: number | null
}

/** Analyzes a trade and returns pre/post scores (or points differentials) for both teams. */
export async function analyzeTrade(
    sessionId: string
    , req: {
        player_assignments: Record<string, number[]>
        my_team: string
        their_team: string
        my_trade: number[]
        their_trade: number[]
        ignore_position_check?: boolean
    }
): Promise<TradeAnalyzeResponse> {
    return jsonRequest(`${BASE_URL}/sessions/${sessionId}/trade/analyze`, 'Trade analyze', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(req),
    })
}

// ── POST /sessions/{id}/trade/suggest ────────────────────────────────────────

export interface TradeSuggestion {
    send: number[]
    receive: number[]
    your_score: number
    their_score: number
    // Points mode fields
    partner_team_id?: string
    user_sends?: number[]
    partner_sends?: number[]
    user_gain?: number
    partner_gain?: number
    total_gain?: number
    trade_format?: string
    user_delta_fp?: number
    partner_delta_fp?: number
    analysis?: TradeAnalyzeResponse
}

export interface TradeSuggestResponse {
    suggestions: TradeSuggestion[]
    scoring_system?: 'categories' | 'points'
}

/** Generates trade suggestions for two teams. May take 10-30s. */
export async function suggestTrades(
    sessionId: string
    , req: {
        player_assignments: Record<string, number[]>
        my_team: string
        their_team: string
        combo_params: { n_traded: number; n_received: number; threshold: number }[]
        your_differential_threshold: number
        their_differential_threshold: number
        ignore_position_check?: boolean
    }
): Promise<TradeSuggestResponse> {
    return jsonRequest(`${BASE_URL}/sessions/${sessionId}/trade/suggest`, 'Trade suggest', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(req),
    })
}

// ── POST /sessions/{id}/streamer/optimize ────────────────────────────────────

/** Optimizes weekly streaming acquisitions and churn schedules for Points mode. */
export async function optimizeStreamer(
    sessionId: string
    , req: {
        player_assignments: Record<string, number[]>
        my_team_id: string
        horizon_days?: number
        max_acquisitions?: number
        mode?: string
        streaming_slots?: number
        designated_drops?: number[]
        candidate_limit?: number
    }
): Promise<StreamerOptimizeResponse> {
    return jsonRequest(`${BASE_URL}/sessions/${sessionId}/streamer/optimize`, 'Streamer optimize', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(req),
    })
}

// ── Platform integration (live: ESPN / Yahoo / Fantrax) ───────────────────────
// `platform` is the exact label from PLATFORM_OPTIONS (e.g. 'Retrieve from Fantrax').

export interface PlatformDivision {
    name: string
    id: string
}

export interface PlatformConnectResponse {
    team_names: string[]
    n_drafters: number
    n_picks: number
    available_modes: string[]
}

export interface DraftStateResponse {
    player_assignments: Record<string, number[]>
    injured_players: number[]
    status: string
    remaining_cash?: Record<string, number>   // Auction Mode only
}

export interface PlatformLeague {
    id: string
    name: string
    season?: number
}

/** Lists a platform league's divisions (empty when the league has none). */
export async function fetchDivisions(
    platform: string
    , leagueId: string
): Promise<PlatformDivision[]> {
    const data = await jsonRequest<{ divisions: PlatformDivision[] }>(
        `${BASE_URL}/platforms/${encodeURIComponent(platform)}/divisions?league_id=${encodeURIComponent(leagueId)}`
        , 'Platform divisions'
    )
    return data.divisions
}

/** Lists the user's leagues for an auth-based platform (empty for manual-id platforms).
 *  Credentials are resolved server-side from the signed-in user's session cookie. */
export async function fetchLeagues(platform: string): Promise<PlatformLeague[]> {
    const data = await jsonRequest<{ leagues: PlatformLeague[] }>(
        `${BASE_URL}/platforms/${encodeURIComponent(platform)}/leagues`
        , 'Platform leagues'
    )
    return data.leagues
}

/** Returns the Yahoo OAuth authorization URL for the user to visit. */
export async function fetchYahooAuthUrl(): Promise<string> {
    const data = await jsonRequest<{ auth_url: string }>(`${BASE_URL}/platforms/yahoo/auth-url`, 'Yahoo auth URL')
    return data.auth_url
}

/** Exchanges a pasted Yahoo authorization code for tokens, persisted for the signed-in user. */
export async function submitYahooToken(authCode: string): Promise<void> {
    await jsonRequest<void>(`${BASE_URL}/platforms/yahoo/token`, 'Yahoo token', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ auth_code: authCode }),
    })
}

/** Persists the signed-in user's ESPN s2 + SWID cookies. */
export async function submitEspnCredentials(s2: string, swid: string): Promise<void> {
    await jsonRequest<void>(`${BASE_URL}/platforms/espn/credentials`, 'ESPN credentials', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ s2, swid }),
    })
}

/** Connects to a platform league and returns its team/shape metadata. */
export async function connectPlatform(
    platform: string
    , leagueId: string
    , divisionId: string | null
): Promise<PlatformConnectResponse> {
    return jsonRequest(
        `${BASE_URL}/platforms/${encodeURIComponent(platform)}/connect`
        , 'Platform connect'
        , {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify({ league_id: leagueId, division_id: divisionId }),
        }
    )
}

/** Polls the current draft/roster state from the connected platform. */
export async function fetchDraftState(
    sessionId: string
    , mode: string
): Promise<DraftStateResponse> {
    return jsonRequest(
        `${BASE_URL}/sessions/${sessionId}/draft-state?mode=${encodeURIComponent(mode)}`
        , 'Draft state'
    )
}
