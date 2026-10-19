// data_entry/season/season_trading.ts
// Renders the Season → Trading tab.
// Replicates the structure of src/tabs/trading.py:
//   1. Team selectors (your team / counterparty)
//   2. Player multi-selects (send / receive) + result sub-tabs (H-score / G-score)
//   3. Suggested trades table

import { makeCustomSelect } from '../../custom_select.js'
import { makeMultiSelectWidget, MultiSelectWidget, makeNumberInput, makeSidebarToggle, readRequiredIntInput } from '../../helper_functions.js'
import { readTeamNames, readRosterAssignments } from './season_helpers.js'
import { getGScoreById, isPointsMode } from '../../app_state.js'
import { getRegistryEntry } from '../../player_registry.js'
import { buildMinimalPlayerDisplayHtml, buildFullPlayerDisplayHtml, buildPlayerOptionLabel } from '../../player_display.js'
import { getSelectedCategories } from '../../parameter_collection/format_and_categories.js'
import { stat_styler_primary } from '../../styles/styler_functions.js'
import { DEFAULT_COMBOS, POINTS_COMBOS } from '../../parameter_collection/trade_parameters.js'
import { pref, savePref } from '../../preferences.js'
import { runTradeAnalyze, runTradeSuggest } from '../../api/season_session.js'
import { applyIndicatorState } from '../../api/session.js'
import type { TradeAnalyzeResponse, TradeSuggestion } from '../../api/client.js'
import type { LeagueTradeImpact } from '../../types.js'

// ─── Player id <-> multiselect value helpers ─────────────────────────────────

/** Multiselect options for a roster: player ids as values, registry names as labels. */
function buildPlayerOptions(playerIds: number[]): { value: string; label: string }[] {
    return playerIds.map(playerId => ({
        value: String(playerId),
        label: buildPlayerOptionLabel(playerId),
        html:  buildFullPlayerDisplayHtml(playerId),
    }))
}

/** Parses multiselect values back to player ids. Throws on a non-numeric value — the
 *  send/receive selectors only ever carry stringified ids. */
function parseSelectedPlayerIds(values: string[]): number[] {
    return values.map(value => {
        const playerId = Number(value)
        if (Number.isNaN(playerId)) throw new Error(`Trade multiselect carried a non-numeric value: "${value}"`)
        return playerId
    })
}

// ─── G-score comparison table ────────────────────────────────────────────────

/** Builds a G-score comparison table for the selected send/receive players. */
function buildGScoreTable(sent: number[], received: number[]): HTMLElement {
    const container = document.createElement('div')
    container.className = 'trade-gscore-section'

    const gScoreMap = getGScoreById()
    const categories = getSelectedCategories()

    type Row = { label: string; total: number; values: number[] }

    function getPlayerGRow(playerId: number): Row | null {
        const gs = gScoreMap.get(playerId)
        if (!gs) return null
        return { label: getRegistryEntry(playerId).name, total: gs.total, values: gs.values }
    }

    const sentRows:     Row[] = []
    const receivedRows: Row[] = []
    for (const playerId of sent) {
        const r = getPlayerGRow(playerId)
        if (r) sentRows.push(r)
    }
    for (const playerId of received) {
        const r = getPlayerGRow(playerId)
        if (r) receivedRows.push(r)
    }

    function sumRows(rows: Row[]): Row {
        const values = categories.map((_, i) => rows.reduce((s, r) => s + r.values[i], 0))
        return { label: '', total: values.reduce((a, b) => a + b, 0), values }
    }

    const sentTotal = sumRows(sentRows)
    const recvTotal = sumRows(receivedRows)
    const diff: Row = {
        label: 'Total Difference',
        total: recvTotal.total - sentTotal.total,
        values: categories.map((_, i) => recvTotal.values[i] - sentTotal.values[i]),
    }

    // Render table
    const table = document.createElement('table')
    table.className = 'trade-result-table'

    const thead = table.createTHead()
    const hrow = thead.insertRow()
    hrow.insertCell().textContent = ''
    const totalTh = document.createElement('th')
    totalTh.textContent = 'Total'
    hrow.append(totalTh)
    for (const cat of categories) {
        const th = document.createElement('th')
        th.textContent = cat
        hrow.append(th)
    }

    const tbody = table.createTBody()

    function addRow(label: string, row: Row, cssClass: string, useGradient: boolean): void {
        const tr = tbody.insertRow()
        tr.className = cssClass
        const labelCell = tr.insertCell()
        labelCell.textContent = label
        labelCell.className = 'trade-row-label'
        const totalCell = tr.insertCell()
        totalCell.textContent = row.total.toFixed(2)
        if (useGradient) {
            totalCell.style.cssText = stat_styler_primary(row.total, 60, 0)
        } else {
            totalCell.className = 'celltypeb'
        }
        for (const v of row.values) {
            const td = tr.insertCell()
            td.textContent = v.toFixed(2)
            if (useGradient) {
                td.style.cssText = stat_styler_primary(v, 60, 0)
            } else {
                td.className = 'celltypeb'
            }
        }
    }

    for (const row of sentRows) {
        addRow(row.label, row, 'trade-row-sent', true)
    }
    addRow('Total Sent', sentTotal, 'trade-row-subtotal', false)

    for (const row of receivedRows) {
        addRow(row.label, row, 'trade-row-received', true)
    }
    addRow('Total Received', recvTotal, 'trade-row-subtotal', false)

    addRow('Total Difference', diff, 'trade-row-diff', true)

    container.append(table)
    return container
}

// ─── H-score trade analysis ─────────────────────────────────────────────────

/** Builds the H-score trade result display with pre/post comparison for both teams. */
function buildHScoreResult(
    pane: HTMLElement
    , assignments: Record<string, number[]>
    , yourTeam: string
    , theirTeam: string
    , sent: number[]
    , received: number[]
): void {
    pane.innerHTML = ''

    const loading = document.createElement('div')
    loading.className = 'trade-hscore-stub'
    loading.textContent = 'Analyzing trade...'
    pane.append(loading)

    const ignore_position_check = (document.getElementById('ts-ignore-position') as HTMLInputElement).checked
    runTradeAnalyze(assignments, yourTeam, theirTeam, sent, received, ignore_position_check)
        .then(resp => renderHScoreResult(pane, resp))
        .catch(err => {
            pane.innerHTML = ''
            const msg = document.createElement('div')
            msg.className = 'trade-hscore-stub'
            msg.textContent = `Error: ${err.message}`
            pane.append(msg)
        })
}

function renderHScoreResult(pane: HTMLElement, resp: TradeAnalyzeResponse): void {
    pane.innerHTML = ''

    if (resp.error) {
        const msg = document.createElement('div')
        msg.className = 'trade-hscore-stub'
        msg.textContent = resp.error
        pane.append(msg)
        return
    }

    if (!resp.your_team || !resp.their_team) return

    const categories = getSelectedCategories()
    const yourImproved = resp.your_team.post.h_score > resp.your_team.pre.h_score
    const theirImproved = resp.their_team.post.h_score > resp.their_team.pre.h_score

    // Sub-tabs: Your Team / Their Team
    const tabNav = document.createElement('div')
    tabNav.className = 'trade-tab-nav'

    const yourBtn = document.createElement('button')
    yourBtn.type = 'button'
    yourBtn.className = 'trade-tab-btn active'
    yourBtn.textContent = `Your Team ${yourImproved ? '\u{1F44D}' : '\u{1F44E}'}`

    const theirBtn = document.createElement('button')
    theirBtn.type = 'button'
    theirBtn.className = 'trade-tab-btn'
    theirBtn.textContent = `Their Team ${theirImproved ? '\u{1F44D}' : '\u{1F44E}'}`

    tabNav.append(yourBtn, theirBtn)
    pane.append(tabNav)

    const yourPane = document.createElement('div')
    const theirPane = document.createElement('div')
    theirPane.style.display = 'none'
    pane.append(yourPane, theirPane)

    yourBtn.addEventListener('click', () => {
        yourBtn.classList.add('active');  theirBtn.classList.remove('active')
        yourPane.style.display = '';      theirPane.style.display = 'none'
    })
    theirBtn.addEventListener('click', () => {
        theirBtn.classList.add('active'); yourBtn.classList.remove('active')
        theirPane.style.display = '';     yourPane.style.display = 'none'
    })

    yourPane.append(buildHScoreComparisonTable(resp.your_team.pre, resp.your_team.post, categories))
    theirPane.append(buildHScoreComparisonTable(resp.their_team.pre, resp.their_team.post, categories))
}

function buildHScoreComparisonTable(
    pre: { h_score: number; rates: number[] }
    , post: { h_score: number; rates: number[] }
    , categories: string[]
): HTMLTableElement {
    const table = document.createElement('table')
    table.className = 'trade-result-table'

    const thead = table.createTHead()
    const hrow = thead.insertRow()
    hrow.insertCell().textContent = ''
    const hTh = document.createElement('th')
    hTh.textContent = 'H-score'
    hrow.append(hTh)
    for (const cat of categories) {
        const th = document.createElement('th')
        th.textContent = cat
        hrow.append(th)
    }

    const tbody = table.createTBody()

    function addRow(label: string, hScore: number, rates: number[]): void {
        const tr = tbody.insertRow()
        const labelCell = tr.insertCell()
        labelCell.textContent = label
        labelCell.className = 'trade-row-label'

        const hCell = tr.insertCell()
        hCell.textContent = (hScore * 100).toFixed(2) + '%'
        hCell.style.cssText = stat_styler_primary(hScore - 0.5, 200, 0)

        for (const rate of rates) {
            const td = tr.insertCell()
            td.textContent = (rate * 100).toFixed(1) + '%'
            td.style.cssText = stat_styler_primary(rate - 0.5, 200, 0)
        }
    }

    addRow('Pre-trade', pre.h_score, pre.rates)
    addRow('Post-trade', post.h_score, post.rates)

    return table
}

// ─── Points mode trade analysis ─────────────────────────────────────────────

/** Builds the Points-mode trade result display with net delta FP and league impact. */
function buildPointsTradeResult(
    mainPane: HTMLElement
    , leaguePane: HTMLElement
    , assignments: Record<string, number[]>
    , yourTeam: string
    , theirTeam: string
    , sent: number[]
    , received: number[]
): void {
    mainPane.innerHTML = ''
    leaguePane.innerHTML = ''

    const loading = document.createElement('div')
    loading.className = 'trade-hscore-stub'
    loading.textContent = 'Analyzing points trade...'
    mainPane.append(loading)

    const ignore_position_check = (document.getElementById('ts-ignore-position') as HTMLInputElement).checked
    runTradeAnalyze(assignments, yourTeam, theirTeam, sent, received, ignore_position_check)
        .then(resp => renderPointsTradeResult(mainPane, leaguePane, resp, yourTeam, theirTeam))
        .catch(err => {
            mainPane.innerHTML = ''
            const msg = document.createElement('div')
            msg.className = 'trade-hscore-stub'
            msg.textContent = `Error: ${err.message}`
            mainPane.append(msg)
        })
}

function renderPointsTradeResult(
    mainPane: HTMLElement
    , leaguePane: HTMLElement
    , resp: TradeAnalyzeResponse
    , yourTeam: string
    , theirTeam: string
): void {
    mainPane.innerHTML = ''
    leaguePane.innerHTML = ''

    if (resp.error || resp.is_legal === false) {
        const msg = document.createElement('div')
        msg.className = 'trade-hscore-stub'
        msg.style.color = '#e05555'
        msg.textContent = resp.rejection_reason || resp.error || 'This trade is impermissible.'
        mainPane.append(msg)
        return
    }

    const yourNet = resp.team_a_net_delta_fp ?? resp.your_net_delta_fp ?? 0
    const theirNet = resp.team_b_net_delta_fp ?? resp.their_net_delta_fp ?? 0
    const isWinWin = yourNet >= 0 && theirNet >= 0

    // Top status banner
    const banner = document.createElement('div')
    banner.style.display = 'flex'
    banner.style.alignItems = 'center'
    banner.style.justifyContent = 'space-between'
    banner.style.padding = '8px 12px'
    banner.style.marginBottom = '12px'
    banner.style.borderRadius = '6px'
    banner.style.backgroundColor = isWinWin ? 'rgba(76, 175, 80, 0.12)' : 'rgba(33, 150, 243, 0.08)'
    banner.style.border = `1px solid ${isWinWin ? '#4caf50' : '#2196f3'}`

    const bannerTitle = document.createElement('span')
    bannerTitle.style.fontWeight = '600'
    bannerTitle.style.fontSize = '0.88rem'
    bannerTitle.textContent = isWinWin ? '🤝 Win-Win Trade Proposal' : '⚖️ Asymmetric Trade Proposal'

    const bannerSummary = document.createElement('span')
    bannerSummary.style.fontSize = '0.82rem'
    bannerSummary.style.color = isWinWin ? '#2e7d32' : 'inherit'
    bannerSummary.textContent = `Your Net: ${yourNet >= 0 ? '+' : ''}${yourNet.toFixed(1)} FP | Their Net: ${theirNet >= 0 ? '+' : ''}${theirNet.toFixed(1)} FP`

    banner.append(bannerTitle, bannerSummary)
    mainPane.append(banner)

    // Points comparison table
    const table = document.createElement('table')
    table.className = 'trade-result-table'

    const thead = table.createTHead()
    const hrow = thead.insertRow()
    const cols = ['Team', 'Pre-Trade FP', 'Player Swap Δ', 'Streamer Spot Boost', 'Post-Trade Net FP', 'Net Delta FP']
    for (const c of cols) {
        const th = document.createElement('th')
        th.textContent = c
        if (c === 'Team') th.style.textAlign = 'left'
        hrow.append(th)
    }

    const tbody = table.createTBody()

    function addTeamRow(
        teamLabel: string
        , preFP: number
        , swapDelta: number
        , openedSpots: number
        , streamerBoost: number
        , postFP: number
        , netDelta: number
    ): void {
        const tr = tbody.insertRow()

        const teamCell = tr.insertCell()
        teamCell.textContent = teamLabel
        teamCell.className = 'trade-row-label'

        const preCell = tr.insertCell()
        preCell.textContent = preFP.toFixed(1)
        preCell.className = 'celltypeb'

        const swapCell = tr.insertCell()
        swapCell.textContent = `${swapDelta >= 0 ? '+' : ''}${swapDelta.toFixed(1)} FP`
        swapCell.className = 'celltypeb'

        const boostCell = tr.insertCell()
        if (openedSpots > 0 || streamerBoost > 0) {
            boostCell.textContent = `+${streamerBoost.toFixed(1)} FP (${openedSpots} open spot${openedSpots > 1 ? 's' : ''})`
            boostCell.style.color = '#2e7d32'
            boostCell.style.fontWeight = '600'
        } else {
            boostCell.textContent = '—'
            boostCell.className = 'celltypeb'
        }

        const postCell = tr.insertCell()
        postCell.textContent = postFP.toFixed(1)
        postCell.className = 'celltypeb'

        const netCell = tr.insertCell()
        netCell.textContent = `${netDelta >= 0 ? '+' : ''}${netDelta.toFixed(1)} FP`
        netCell.style.cssText = stat_styler_primary(netDelta, 60, 0)
        netCell.style.fontWeight = '700'
    }

    addTeamRow(
        `Your Team (${yourTeam})`,
        resp.team_a_pre_fp ?? 0,
        resp.team_a_delta_fp ?? 0,
        resp.team_a_opened_spots ?? 0,
        resp.team_a_streamer_boost ?? 0,
        resp.team_a_post_fp ?? 0,
        yourNet,
    )

    addTeamRow(
        `Their Team (${theirTeam})`,
        resp.team_b_pre_fp ?? 0,
        resp.team_b_delta_fp ?? 0,
        resp.team_b_opened_spots ?? 0,
        resp.team_b_streamer_boost ?? 0,
        resp.team_b_post_fp ?? 0,
        theirNet,
    )

    mainPane.append(table)

    // Streamer Spot Valuation Explanation Callout
    const openedTotal = (resp.team_a_opened_spots ?? 0) + (resp.team_b_opened_spots ?? 0)
    if (openedTotal > 0) {
        const streamNote = document.createElement('div')
        streamNote.style.marginTop = '12px'
        streamNote.style.padding = '8px 12px'
        streamNote.style.borderRadius = '4px'
        streamNote.style.fontSize = '0.78rem'
        streamNote.style.backgroundColor = 'light-dark(#f5f9f6, #1a2a22)'
        streamNote.style.border = '1px solid light-dark(#c8e6c9, #2d5a3f)'
        streamNote.style.color = 'light-dark(#2e5939, #a5d6a7)'

        const streamBoost = (resp.team_a_opened_spots ?? 0) > 0 ? (resp.team_a_streamer_boost ?? 0) : (resp.team_b_streamer_boost ?? 0)
        const teamBenefiting = (resp.team_a_opened_spots ?? 0) > 0 ? yourTeam : theirTeam
        streamNote.innerHTML = `<strong>💡 Streamer Spot Valuation:</strong> When consolidating talent in an asymmetric trade (e.g. 2v1), an open roster spot is created. This spot can be streamed across ~6.5 waiver games/week at baseline, contributing an estimated <strong>+${streamBoost.toFixed(1)} FP</strong> to ${teamBenefiting}.`
        mainPane.append(streamNote)
    }

    // Populate League Impact pane
    if (resp.league_impact) {
        renderLeagueImpact(leaguePane, resp.league_impact, yourTeam, theirTeam)
    } else {
        const noLeague = document.createElement('div')
        noLeague.className = 'trade-hscore-stub'
        noLeague.textContent = 'To view full league standings impact and competitive balance metrics, populate all league rosters on the Rosters tab.'
        leaguePane.append(noLeague)
    }
}

function renderLeagueImpact(
    pane: HTMLElement
    , impact: LeagueTradeImpact
    , yourTeam: string
    , theirTeam: string
): void {
    pane.innerHTML = ''

    // Parity / Competitive Balance Summary Cards
    const summaryRow = document.createElement('div')
    summaryRow.style.display = 'grid'
    summaryRow.style.gridTemplateColumns = 'repeat(auto-fit, minmax(140px, 1fr))'
    summaryRow.style.gap = '8px'
    summaryRow.style.marginBottom = '12px'

    function makeMetricCard(label: string, value: string, subtext?: string): HTMLElement {
        const card = document.createElement('div')
        card.style.padding = '8px 10px'
        card.style.borderRadius = '6px'
        card.style.border = '1px solid light-dark(#e0e0e0, #333355)'
        card.style.backgroundColor = 'light-dark(#fafafa, #1e1e2f)'

        const l = document.createElement('div')
        l.style.fontSize = '0.72rem'
        l.style.color = 'light-dark(#666, #99a)'
        l.textContent = label

        const v = document.createElement('div')
        v.style.fontSize = '0.95rem'
        v.style.fontWeight = '700'
        v.style.marginTop = '2px'
        v.textContent = value

        card.append(l, v)
        if (subtext) {
            const s = document.createElement('div')
            s.style.fontSize = '0.68rem'
            s.style.color = 'light-dark(#888, #778)'
            s.textContent = subtext
            card.append(s)
        }
        return card
    }

    summaryRow.append(
        makeMetricCard('Pre-Trade Spread', `${impact.pre_trade_spread.toFixed(1)} FP`, '1st vs Last place'),
        makeMetricCard('Post-Trade Spread', `${impact.post_trade_spread.toFixed(1)} FP`, '1st vs Last place'),
        makeMetricCard('Competitive Balance Δ', `${impact.competitive_balance_delta >= 0 ? '+' : ''}${impact.competitive_balance_delta.toFixed(2)}`, 'Std dev change'),
    )
    pane.append(summaryRow)

    // Standings Impact Table
    const table = document.createElement('table')
    table.className = 'trade-result-table'

    const thead = table.createTHead()
    const hrow = thead.insertRow()
    const headers = ['Rank', 'Team', 'Pre FP', 'Post FP', 'Net Δ FP', 'Rank Shift']
    for (const h of headers) {
        const th = document.createElement('th')
        th.textContent = h
        if (h === 'Team') th.style.textAlign = 'left'
        hrow.append(th)
    }

    const tbody = table.createTBody()

    // Get all teams and sort by post-trade points descending
    const teams = Object.keys(impact.post_trade_points).sort((a, b) => {
        return (impact.post_trade_points[b] ?? 0) - (impact.post_trade_points[a] ?? 0)
    })

    for (let idx = 0; idx < teams.length; idx++) {
        const teamName = teams[idx]
        const tr = tbody.insertRow()

        const isTraded = teamName === yourTeam || teamName === theirTeam
        if (isTraded) {
            tr.style.backgroundColor = teamName === yourTeam ? 'rgba(33, 150, 243, 0.08)' : 'rgba(255, 152, 0, 0.08)'
        }

        const rankCell = tr.insertCell()
        const postRank = impact.post_trade_ranks[teamName] ?? (idx + 1)
        rankCell.textContent = `#${postRank}`
        rankCell.style.fontWeight = '600'

        const nameCell = tr.insertCell()
        nameCell.className = 'trade-row-label'
        nameCell.textContent = teamName + (teamName === yourTeam ? ' (You)' : teamName === theirTeam ? ' (Partner)' : '')

        const preFp = impact.pre_trade_points[teamName] ?? 0
        const postFp = impact.post_trade_points[teamName] ?? 0
        const delta = postFp - preFp
        const shift = impact.rank_shifts[teamName] ?? 0

        const preCell = tr.insertCell()
        preCell.textContent = preFp.toFixed(1)
        preCell.className = 'celltypeb'

        const postCell = tr.insertCell()
        postCell.textContent = postFp.toFixed(1)
        postCell.className = 'celltypeb'

        const deltaCell = tr.insertCell()
        deltaCell.textContent = `${delta >= 0 ? '+' : ''}${delta.toFixed(1)}`
        if (Math.abs(delta) > 0.01) {
            deltaCell.style.cssText = stat_styler_primary(delta, 60, 0)
            deltaCell.style.fontWeight = '600'
        } else {
            deltaCell.className = 'celltypeb'
        }

        const shiftCell = tr.insertCell()
        if (shift > 0) {
            shiftCell.textContent = `▲ +${shift}`
            shiftCell.style.color = '#2e7d32'
            shiftCell.style.fontWeight = '600'
        } else if (shift < 0) {
            shiftCell.textContent = `▼ ${shift}`
            shiftCell.style.color = '#c62828'
            shiftCell.style.fontWeight = '600'
        } else {
            shiftCell.textContent = '—'
            shiftCell.style.color = 'light-dark(#999, #778)'
        }
    }

    pane.append(table)
}

// ─── Suggested trades ───────────────────────────────────────────────────────

/**
 * Rebuilds the suggestion display from whatever is currently in the cache.
 * Shows a loading indicator for any combos that are still being fetched,
 * and merges + sorts all cached suggestions into a single table.
 */
function refreshSuggestionDisplay(
    resultsArea: HTMLElement
    , selected: string[]
    , pendingFetches: Set<string>
    , suggestionCache: Map<string, TradeSuggestion[]>
    , sendSel: MultiSelectWidget
    , receiveSel: MultiSelectWidget
    , onTradeSelected: () => void
): void {
    resultsArea.innerHTML = ''

    if (selected.length === 0) {
        applyIndicatorState('suggest-indicator', 'idle')
        const msg = document.createElement('div')
        msg.className = 'trade-suggestion-stub'
        msg.textContent = 'Select at least one trade combination.'
        resultsArea.append(msg)
        return
    }

    const isLoading = selected.some(key => pendingFetches.has(key))
    if (!isLoading) applyIndicatorState('suggest-indicator', 'idle')

    const allSuggestions = selected.flatMap(key => suggestionCache.get(key) ?? [])
    allSuggestions.sort((a, b) => {
        const scoreA = a.user_delta_fp ?? a.user_gain ?? a.your_score ?? 0
        const scoreB = b.user_delta_fp ?? b.user_gain ?? b.your_score ?? 0
        return scoreB - scoreA
    })

    if (allSuggestions.length > 0) {
        resultsArea.append(buildSuggestionTable(allSuggestions, sendSel, receiveSel, onTradeSelected))
    } else if (!isLoading) {
        const msg = document.createElement('div')
        msg.className = 'trade-suggestion-stub'
        msg.textContent = 'No promising trades found.'
        resultsArea.append(msg)
    }
}

/**
 * Fetches any selected combos not yet in the cache, then refreshes the display.
 * Already-cached combos are shown immediately without a round trip.
 */
function fetchMissingCombos(
    resultsArea: HTMLElement
    , comboSel: MultiSelectWidget
    , assignments: Record<string, number[]>
    , yourTeam: string
    , theirTeam: string
    , pendingFetches: Set<string>
    , suggestionCache: Map<string, TradeSuggestion[]>
    , sendSel: MultiSelectWidget
    , receiveSel: MultiSelectWidget
    , onTradeSelected: () => void
): void {
    const points = isPointsMode()
    const your_differential_threshold  = points
        ? (parseFloat((document.getElementById('ts-your-threshold')  as HTMLInputElement).value) || 0)
        : (parseFloat((document.getElementById('ts-your-threshold')  as HTMLInputElement).value) || 0) / 100
    const their_differential_threshold = points
        ? (parseFloat((document.getElementById('ts-their-threshold') as HTMLInputElement).value) || 0)
        : (parseFloat((document.getElementById('ts-their-threshold') as HTMLInputElement).value) || 0) / 100
    const ignore_position_check        = (document.getElementById('ts-ignore-position') as HTMLInputElement).checked
    const selected = comboSel.getSelected()
    const availableCombos = points ? POINTS_COMBOS : DEFAULT_COMBOS

    for (const key of selected) {
        if (suggestionCache.has(key) || pendingFetches.has(key)) continue

        const cp = availableCombos.find(p => `${p.n_traded} for ${p.n_received}` === key)
        if (!cp) continue

        pendingFetches.add(key)

        runTradeSuggest(
            assignments, yourTeam, theirTeam,
            [cp], your_differential_threshold, their_differential_threshold,
            ignore_position_check,
        )
            .then(resp => {
                pendingFetches.delete(key)
                suggestionCache.set(key, resp.suggestions)
                if (pendingFetches.size === 0) {
                    refreshSuggestionDisplay(resultsArea, comboSel.getSelected(), pendingFetches, suggestionCache, sendSel, receiveSel, onTradeSelected)
                }
            })
            .catch(() => {
                pendingFetches.delete(key)
                suggestionCache.set(key, [])
                if (pendingFetches.size === 0) {
                    refreshSuggestionDisplay(resultsArea, comboSel.getSelected(), pendingFetches, suggestionCache, sendSel, receiveSel, onTradeSelected)
                }
            })
    }

    refreshSuggestionDisplay(resultsArea, selected, pendingFetches, suggestionCache, sendSel, receiveSel, onTradeSelected)
}

function buildSuggestionTable(
    suggestions: TradeSuggestion[]
    , sendSel: MultiSelectWidget
    , receiveSel: MultiSelectWidget
    , onTradeSelected: () => void
): HTMLTableElement {
    const table = document.createElement('table')
    table.className = 'trade-result-table'

    const points = isPointsMode()
    const thead = table.createTHead()
    const hrow = thead.insertRow()
    const headers = points
        ? ['Send', 'Receive', 'Your Delta FP', 'Their Delta FP', 'Format']
        : ['Send', 'Receive', 'Your Score', 'Their Score']

    for (const header of headers) {
        const th = document.createElement('th')
        th.textContent = header
        if (header === 'Send' || header === 'Receive') th.style.textAlign = 'left'
        hrow.append(th)
    }

    const tbody = table.createTBody()

    for (const sug of suggestions) {
        const tr = tbody.insertRow()
        tr.style.cursor = 'pointer'

        const sendIds = sug.send ?? sug.user_sends ?? []
        const recvIds = sug.receive ?? sug.partner_sends ?? []

        const sendCell = tr.insertCell()
        sendCell.className = 'trade-suggestion-players'
        sendCell.innerHTML = sendIds.map(buildMinimalPlayerDisplayHtml).join('')
        sendCell.style.textAlign = 'left'

        const recvCell = tr.insertCell()
        recvCell.className = 'trade-suggestion-players'
        recvCell.innerHTML = recvIds.map(buildMinimalPlayerDisplayHtml).join('')
        recvCell.style.textAlign = 'left'

        const yourCell = tr.insertCell()
        const theirCell = tr.insertCell()

        if (points) {
            const yourVal = sug.user_delta_fp ?? sug.user_gain ?? sug.your_score ?? 0
            const theirVal = sug.partner_delta_fp ?? sug.partner_gain ?? sug.their_score ?? 0

            yourCell.textContent = `${yourVal >= 0 ? '+' : ''}${yourVal.toFixed(1)} FP`
            yourCell.style.cssText = stat_styler_primary(yourVal, 60, 0)

            theirCell.textContent = `${theirVal >= 0 ? '+' : ''}${theirVal.toFixed(1)} FP`
            theirCell.style.cssText = stat_styler_primary(theirVal, 60, 0)

            const formatCell = tr.insertCell()
            formatCell.textContent = sug.trade_format || `${sendIds.length}v${recvIds.length}`
            formatCell.className = 'celltypeb'
        } else {
            const yourScore = sug.your_score ?? 0
            const theirScore = sug.their_score ?? 0

            yourCell.textContent = (yourScore * 100).toFixed(2) + '%'
            yourCell.style.cssText = stat_styler_primary(yourScore, 15000, 0)

            theirCell.textContent = (theirScore * 100).toFixed(2) + '%'
            theirCell.style.cssText = stat_styler_primary(theirScore, 15000, 0)
        }

        // Clicking a suggestion populates the send/receive selectors.
        // Both are set silently to avoid two separate updateResults() calls
        // (which would fire a stale API request on the first onChange).
        tr.addEventListener('click', () => {
            sendSel.setSelectedSilently(sendIds.map(String))
            receiveSel.setSelectedSilently(recvIds.map(String))
            onTradeSelected()
        })
    }

    return table
}

// Tracks the listeners attached by the most recent renderSeasonTrading call so
// they can be detached before the next one. Without this, switching tabs in and
// out of Trading would leave the previous render's custom selects and
// multiselect widgets' internal listeners bound to detached nodes.
let tradingListenerController: AbortController | null = null

// ─── Main render ─────────────────────────────────────────────────────────────

/** Renders the full Trading tab into the given container element. */
export function renderSeasonTrading(container: HTMLElement): void {
    tradingListenerController?.abort()
    tradingListenerController = new AbortController()

    container.innerHTML = ''

    const teamNames   = readTeamNames()
    const assignments = readRosterAssignments()

    const nPicks = readRequiredIntInput('ls-n-picks')
    const fullTeams = teamNames.filter(t => (assignments[t]?.length ?? 0) >= nPicks)

    if (fullTeams.length < 2) {
        const msg = document.createElement('p')
        msg.className = 'coming-soon'
        msg.textContent = 'Fill out at least two full teams on the Rosters tab to use Trading.'
        container.append(msg)
        return
    }

    // ── Row 1: team selectors ────────────────────────────────────────────────

    const selectorRow = document.createElement('div')
    selectorRow.className = 'trade-selector-row'

    // Your team
    const yourWrap = document.createElement('div')
    yourWrap.className = 'trade-selector-col'
    const yourLabel = document.createElement('div')
    yourLabel.className = 'pick-control-label'
    yourLabel.textContent = 'Which team do you want to trade from?'
    yourWrap.append(yourLabel)
    const yourTeamSel = makeCustomSelect(
        'trade-your-team',
        fullTeams.map(n => ({ value: n, label: n })),
        undefined,
        undefined,
        tradingListenerController.signal,
    )
    yourWrap.append(yourTeamSel.element)
    selectorRow.append(yourWrap)

    // Counterparty
    const theirWrap = document.createElement('div')
    theirWrap.className = 'trade-selector-col'
    const theirLabel = document.createElement('div')
    theirLabel.className = 'pick-control-label'
    theirLabel.textContent = 'Which team do you want to trade with?'
    theirWrap.append(theirLabel)
    const counterpartyOptions = fullTeams.filter(n => n !== fullTeams[0])
    const theirTeamSel = makeCustomSelect(
        'trade-their-team',
        counterpartyOptions.map(n => ({ value: n, label: n })),
        undefined,
        undefined,
        tradingListenerController.signal,
    )
    theirWrap.append(theirTeamSel.element)
    selectorRow.append(theirWrap)

    container.append(selectorRow)

    // ── Divider ──────────────────────────────────────────────────────────────

    const divider = document.createElement('hr')
    divider.className = 'trade-divider'
    container.append(divider)

    // ── Row 2: player selects + results (re-rendered on team change) ─────────

    const bodyArea = document.createElement('div')
    container.append(bodyArea)

    // ── Suggested trades (persists across team changes) ──────────────────

    const divider2 = document.createElement('hr')
    divider2.className = 'trade-divider'
    container.append(divider2)

    // Combo filter row — "Suggested trades" label on left, controls on right
    const comboRow = document.createElement('div')
    comboRow.className = 'trade-combo-row'

    const suggestHeader = document.createElement('span')
    suggestHeader.className = 'trade-suggest-header'
    suggestHeader.textContent = 'Suggested trades'

    const suggestIndicator = document.createElement('span')
    suggestIndicator.className = 'eval-indicator'
    suggestIndicator.id = 'suggest-indicator'
    suggestIndicator.dataset.state = 'idle'
    comboRow.append(suggestHeader, suggestIndicator)

    // Combo filter group — right-justified, with "Trade sizes" caption
    const comboGroup = document.createElement('div')
    comboGroup.className = 'trade-threshold-group'
    comboGroup.style.marginLeft = 'auto'

    const comboSizeLabel = document.createElement('span')
    comboSizeLabel.className = 'trade-threshold-label'
    comboSizeLabel.textContent = 'Trade sizes'

    const activeCombos = isPointsMode() ? POINTS_COMBOS : DEFAULT_COMBOS
    const comboOptions = activeCombos.map(cp => `${cp.n_traded} for ${cp.n_received}`)
    const comboSel = makeMultiSelectWidget('', comboOptions.map(combo => ({ value: combo, label: combo })))
    comboSel.element.classList.add('trade-combo-select')
    comboSel.setSelected([comboOptions[0]])

    comboGroup.append(comboSizeLabel, comboSel.element)
    comboRow.append(comboGroup)

    // Your threshold control group
    const yourThreshGroup = document.createElement('div')
    yourThreshGroup.className = 'trade-threshold-group'
    const yourThreshLabel = document.createElement('label')
    yourThreshLabel.htmlFor = 'ts-your-threshold'
    yourThreshLabel.textContent = 'Your threshold'
    const yourThreshInfo = document.createElement('button')
    yourThreshInfo.type = 'button'
    yourThreshInfo.className = 'info-btn'
    yourThreshInfo.textContent = 'ⓘ'
    yourThreshInfo.dataset.tooltip = isPointsMode()
        ? 'Minimum Net Fantasy Points improvement required for a trade to be suggested for your team, in FP. Default: 0.'
        : 'Minimum H-score improvement required for a trade to be suggested for your team, as a percentage. Default: 0.'
    const yourThreshInput = makeNumberInput('ts-your-threshold', pref('ts-your-threshold', 0))
    yourThreshInput.className = ''
    yourThreshInput.step = '0.1'
    const yourThreshWrap = document.createElement('div')
    yourThreshWrap.className = 'trade-threshold-input-wrap'
    yourThreshWrap.append(yourThreshInput)
    yourThreshGroup.append(yourThreshLabel, yourThreshInfo, yourThreshWrap)
    comboRow.append(yourThreshGroup)

    // Their threshold control group
    const theirThreshGroup = document.createElement('div')
    theirThreshGroup.className = 'trade-threshold-group'
    const theirThreshLabel = document.createElement('label')
    theirThreshLabel.htmlFor = 'ts-their-threshold'
    theirThreshLabel.textContent = 'Their threshold'
    const theirThreshInfo = document.createElement('button')
    theirThreshInfo.type = 'button'
    theirThreshInfo.className = 'info-btn'
    theirThreshInfo.textContent = 'ⓘ'
    theirThreshInfo.dataset.tooltip = isPointsMode()
        ? 'Minimum Net Fantasy Points improvement for the counterparty team, in FP. Negative values allow trades that hurt them slightly. Default: 0.'
        : 'Minimum H-score improvement for the counterparty team, as a percentage. Negative values allow trades that hurt them slightly. Default: 0.'
    const theirThreshInput = makeNumberInput('ts-their-threshold', pref('ts-their-threshold', 0))
    theirThreshInput.className = ''
    theirThreshInput.step = '0.1'
    const theirThreshWrap = document.createElement('div')
    theirThreshWrap.className = 'trade-threshold-input-wrap'
    theirThreshWrap.append(theirThreshInput)
    theirThreshGroup.append(theirThreshLabel, theirThreshInfo, theirThreshWrap)
    comboRow.append(theirThreshGroup)

    // Ignore position toggle
    const ignorePosToggle = makeSidebarToggle('ts-ignore-position', 'Ignore position')
    comboRow.append(ignorePosToggle)

    container.append(comboRow)

    const ignorePosInput = document.getElementById('ts-ignore-position') as HTMLInputElement
    ignorePosInput.checked = pref('ts-ignore-position', false)

    const suggestResults = document.createElement('div')
    suggestResults.dataset.testid = 'trade-suggestions'
    container.append(suggestResults)

    // Cache of suggestion results keyed by combo label (e.g. "1 for 1").
    // Cleared when the team pair changes so stale results are never shown.
    const suggestionCache = new Map<string, TradeSuggestion[]>()
    const pendingFetches  = new Set<string>()

    // Track current send/receive selectors and updateResults so combo change can re-trigger search
    let currentSendSel:      MultiSelectWidget | null = null
    let currentReceiveSel:   MultiSelectWidget | null = null
    let currentUpdateResults: (() => void) | null     = null

    function rebuildBody(): void {
        bodyArea.innerHTML = ''
        suggestionCache.clear()
        pendingFetches.clear()

        const yourTeam  = yourTeamSel.getValue() || fullTeams[0]
        const theirTeam = theirTeamSel.getValue() || fullTeams[1]

        if (!yourTeam || !theirTeam) {
            bodyArea.innerHTML = '<p class="coming-soon">Select two different teams.</p>'
            return
        }

        const yourPlayers  = assignments[yourTeam]  ?? []
        const theirPlayers = assignments[theirTeam] ?? []

        if (yourPlayers.length < nPicks || theirPlayers.length < nPicks) {
            bodyArea.innerHTML = '<p class="coming-soon">Both teams must have full rosters.</p>'
            return
        }

        const bodyRow = document.createElement('div')
        bodyRow.className = 'trade-body-row'

        // ── Left: multi-selects ──────────────────────────────────────────────

        const leftCol = document.createElement('div')
        leftCol.className = 'trade-left-col'

        const sendSel    = makeMultiSelectWidget('Which players are you trading?',   buildPlayerOptions(yourPlayers))
        const receiveSel = makeMultiSelectWidget('Which players are you receiving?', buildPlayerOptions(theirPlayers))
        currentSendSel = sendSel
        currentReceiveSel = receiveSel

        leftCol.append(sendSel.element)
        leftCol.append(receiveSel.element)
        bodyRow.append(leftCol)

        // ── Right: result tabs ───────────────────────────────────────────────

        const rightCol = document.createElement('div')
        rightCol.className = 'trade-right-col'

        // Tab buttons
        const tabNav = document.createElement('div')
        tabNav.className = 'trade-tab-nav'
        const hBtn = document.createElement('button')
        hBtn.type = 'button'
        hBtn.className = 'trade-tab-btn active'
        hBtn.textContent = isPointsMode() ? 'Points Analysis' : 'H-score'
        const gBtn = document.createElement('button')
        gBtn.type = 'button'
        gBtn.className = 'trade-tab-btn'
        gBtn.textContent = isPointsMode() ? 'League Impact' : 'G-score'
        tabNav.append(hBtn, gBtn)
        rightCol.append(tabNav)

        // Tab panes
        const hPane = document.createElement('div')
        hPane.className = 'trade-tab-pane'
        hPane.dataset.testid = 'trade-hscore-pane'
        const gPane = document.createElement('div')
        gPane.className = 'trade-tab-pane'
        gPane.dataset.testid = 'trade-gscore-pane'
        gPane.style.display = 'none'
        rightCol.append(hPane, gPane)

        hBtn.addEventListener('click', () => {
            hBtn.classList.add('active');  gBtn.classList.remove('active')
            hPane.style.display = '';      gPane.style.display = 'none'
        })
        gBtn.addEventListener('click', () => {
            gBtn.classList.add('active');  hBtn.classList.remove('active')
            gPane.style.display = '';      hPane.style.display = 'none'
        })

        function updateResults(): void {
            hPane.innerHTML = ''
            gPane.innerHTML = ''

            const sent     = parseSelectedPlayerIds(sendSel.getSelected())
            const received = parseSelectedPlayerIds(receiveSel.getSelected())

            if (sent.length === 0 || received.length === 0) {
                const msg = document.createElement('p')
                msg.className = 'coming-soon'
                msg.textContent = 'A trade must include at least one player from each team.'
                hPane.append(msg)
                gPane.append(msg.cloneNode(true))
                return
            }

            if (!isPointsMode() && sent.length !== received.length) {
                const msg = document.createElement('p')
                msg.className = 'coming-soon'
                msg.textContent = 'Trades must include an equal number of players from both teams.'
                hPane.append(msg)
                gPane.append(msg.cloneNode(true))
                return
            }

            if (isPointsMode()) {
                buildPointsTradeResult(hPane, gPane, assignments, yourTeam, theirTeam, sent, received)
            } else {
                buildHScoreResult(hPane, assignments, yourTeam, theirTeam, sent, received)
                gPane.append(buildGScoreTable(sent, received))
            }
        }

        currentUpdateResults = updateResults
        updateResults()
        sendSel.onChange(updateResults)
        receiveSel.onChange(updateResults)

        bodyRow.append(rightCol)
        bodyArea.append(bodyRow)

        // Fetch suggestions for the current teams
        fetchMissingCombos(suggestResults, comboSel, assignments, yourTeam, theirTeam, pendingFetches, suggestionCache, sendSel, receiveSel, updateResults)
    }

    rebuildBody()

    // When inline threshold controls change, clear the cache and re-fetch.
    // document.contains guard makes this a no-op if the tab has been torn down.
    function clearCacheAndRefetch(): void {
        if (!document.contains(suggestResults)) return
        suggestionCache.clear()
        pendingFetches.clear()
        if (!currentSendSel || !currentReceiveSel || !currentUpdateResults) return
        const yourTeam  = yourTeamSel.getValue() || fullTeams[0]
        const theirTeam = theirTeamSel.getValue() || fullTeams[1]
        fetchMissingCombos(suggestResults, comboSel, assignments, yourTeam, theirTeam, pendingFetches, suggestionCache, currentSendSel, currentReceiveSel, currentUpdateResults)
    }

    yourThreshInput.addEventListener('change', () => {
        savePref('ts-your-threshold', parseFloat(yourThreshInput.value))
        clearCacheAndRefetch()
    })
    theirThreshInput.addEventListener('change', () => {
        savePref('ts-their-threshold', parseFloat(theirThreshInput.value))
        clearCacheAndRefetch()
    })
    ignorePosInput.addEventListener('change', () => {
        savePref('ts-ignore-position', ignorePosInput.checked)
        clearCacheAndRefetch()
    })

    comboSel.onChange(() => {
        if (!currentSendSel || !currentReceiveSel || !currentUpdateResults) return
        const yourTeam  = yourTeamSel.getValue() || fullTeams[0]
        const theirTeam = theirTeamSel.getValue() || fullTeams[1]
        fetchMissingCombos(suggestResults, comboSel, assignments, yourTeam, theirTeam, pendingFetches, suggestionCache, currentSendSel, currentReceiveSel, currentUpdateResults)
    })
    yourTeamSel.element.addEventListener('change', () => {
        const yourTeam = yourTeamSel.getValue()
        theirTeamSel.setOptions(
            fullTeams.filter(n => n !== yourTeam).map(n => ({ value: n, label: n })),
        )
        rebuildBody()
    })
    theirTeamSel.element.addEventListener('change', rebuildBody)
}
