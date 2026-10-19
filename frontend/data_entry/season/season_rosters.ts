// data_entry/season/season_rosters.ts
// Renders the season roster entry table (left) and team selector + G-score
// inspector table (right).  Used by layout.ts for Season → Rosters tab.

import { makeCustomSelect, CustomSelect } from '../../custom_select.js'
import { getPlayerResults, getGScoreById, getShortCategoryNames, isPointsMode, getPlayerResultsById } from '../../app_state.js'
import { getRegistryEntry, findPlayerIdByName } from '../../player_registry.js'
import { makeFullPlayerDisplay, buildFullPlayerDisplayHtml, buildPlayerOptionLabel } from '../../player_display.js'
import { isMobileViewport, readRequiredIntInput } from '../../helper_functions.js'
import { DEFAULT_SEASON_ROSTERS } from './default_season_rosters.js'
import { getTeamLabel, makeTeamLabelInput } from '../team_labels.js'
import { getSelectedCategories, getScoringFormat } from '../../parameter_collection/format_and_categories.js'
import { getLeagueSettings } from '../../parameter_collection/league_settings.js'
import { stat_styler_primary } from '../../styles/styler_functions.js'
import { evaluateTeamHScore, getLivePlatformRosters } from '../../api/season_session.js'
import { getSlotCounts } from '../../parameter_collection/slot_counts.js'
import type { PlayerResult } from '../../types.js'

// Tracks the change-event listeners attached by the most recent render so they
// can be removed before the next one. Without this, calling renderSeasonRosters
// a second time (e.g. revisiting Season Mode) leaves listener closures bound to
// the now-detached previous selects. Each closure captures `selects` (156 custom
// selects × ~300 options) and `rebuildInspector`, so the entire previous render
// tree stays alive in memory. Aborting this controller before the new render
// detaches all listeners at once — no manual iteration, no missed cleanup.
let rosterListenerController: AbortController | null = null

/** Renders the season roster entry grid (left) and team inspector with G-score table (right). */
export function renderSeasonRosters(leftEl: HTMLElement, rightEl: HTMLElement): void {

    // Remove change listeners from any previously-rendered selects so their
    // closures can be garbage-collected. See comment on rosterListenerController.
    rosterListenerController?.abort()
    rosterListenerController = new AbortController()
    const listenerOpts = { signal: rosterListenerController.signal }
    const playerResults = getPlayerResults()
    if (playerResults === null) return

    const nDrafters = readRequiredIntInput('ls-n-drafters')
    const nPicks    = readRequiredIntInput('ls-n-picks')
    const teamNames = (document.getElementById('ls-team-names') as HTMLTextAreaElement)
        .value.split('\n').map(s => s.trim()).filter(Boolean)
    // Option order follows playerResults (G-rank descending, or FP/G descending in points mode),
    // so the dropdowns list the best players first; labels come from the registry.
    const sortedPlayerResults = isPointsMode()
        ? [...playerResults].sort((a, b) => (b.points_per_game ?? b.total_projected_points ?? 0) - (a.points_per_game ?? a.total_projected_points ?? 0))
        : playerResults
    const playerOptions = sortedPlayerResults.map(p => ({
        value: String(p.player_id)
      , label: buildPlayerOptionLabel(p.player_id)
      , html:  buildFullPlayerDisplayHtml(p.player_id)
    }))

    // Live platform → prefill the grid from the platform's rosters (blank until the
    // async poll populates the cache); own data → the hardcoded defaults.
    const isLivePlatform = getLeagueSettings().platform !== 'Enter your own data'
    const platformRosters = isLivePlatform ? getLivePlatformRosters() : null

    leftEl.innerHTML  = ''
    rightEl.innerHTML = ''

    // ── Left: roster entry table ────────────────────────────────────────────

    const scroll = document.createElement('div')
    scroll.className = 'entry-table-scroll'

    const table = document.createElement('table')
    table.className = 'entry-table'

    // Header: Pick | Team1 | Team2 | …
    const thead = table.createTHead()
    const hrow  = thead.insertRow()
    const pickTh = document.createElement('th')
    pickTh.textContent = 'Pick'
    pickTh.style.width = '48px'
    hrow.append(pickTh)
    teamNames.forEach((name, d) => {
        const th = document.createElement('th')
        // Own data → editable display label ("Team N" by default); live platform → the real
        // team name (identity), shown read-only since it comes from the platform.
        if (isLivePlatform) {
            th.textContent = name
        } else {
            th.className = 'team-header-cell'
            const headerWrap = document.createElement('div')
            headerWrap.className = 'team-header'
            headerWrap.append(makeTeamLabelInput(d, rosterListenerController!.signal))
            th.append(headerWrap)
        }
        hrow.append(th)
    })

    // Data rows — one row per pick, one column per team
    const selects: CustomSelect[][] = []   // [row][col]
    const blankOption = [{ value: '', label: '' }]
    const tbody = table.createTBody()

    for (let r = 0; r < nPicks; r++) {
        const row  = tbody.insertRow()
        const rowSelects: CustomSelect[] = []

        const pickCell = row.insertCell()
        pickCell.className   = 'entry-cell-label'
        pickCell.textContent = String(r + 1)

        for (let d = 0; d < nDrafters; d++) {
            const cell = row.insertCell()
            const sel  = makeCustomSelect(
                `sr-player-${r}-${d}`
              , [...blankOption, ...playerOptions]
              , undefined
              , true
              , rosterListenerController.signal
            )
            // Live platform rosters arrive as player ids; DEFAULT_SEASON_ROSTERS stays a
            // name-keyed file and is resolved through the registry at render time. A default
            // name missing from the current pool simply leaves the cell blank.
            if (isLivePlatform) {
                const prefillPlayerId = platformRosters?.[teamNames[d]]?.[r]
                if (prefillPlayerId !== undefined && prefillPlayerId !== -1) {
                    sel.setValue(String(prefillPlayerId))
                }
            } else {
                const defaultName = DEFAULT_SEASON_ROSTERS[teamNames[d]]?.[r]
                if (defaultName) {
                    const prefillPlayerId = resolveRosterEntryName(defaultName)
                    if (prefillPlayerId !== null) sel.setValue(String(prefillPlayerId))
                }
            }
            cell.append(sel.element)
            rowSelects.push(sel)
        }
        selects.push(rowSelects)
    }

    // ── Excel-style cell selection + Copy/Paste ────────────────────────────────

    // Selection range (grid coordinates, not DOM indices)
    let anchorRow = 0, anchorCol = 0
    let focusRow  = 0, focusCol  = 0

    function selectionBounds() {
        return {
            r1: Math.min(anchorRow, focusRow)
          , r2: Math.max(anchorRow, focusRow)
          , c1: Math.min(anchorCol, focusCol)
          , c2: Math.max(anchorCol, focusCol)
        }
    }

    function clearSelectionHighlight(): void {
        table.querySelectorAll('.entry-cell-selected').forEach(el =>
            el.classList.remove('entry-cell-selected')
        )
    }

    function applySelectionHighlight(): void {
        clearSelectionHighlight()
        const { r1, r2, c1, c2 } = selectionBounds()
        for (let r = r1; r <= r2; r++) {
            for (let c = c1; c <= c2; c++) {
                // tbody row r, cell c+1 (cell 0 is the pick label)
                const cell = tbody.rows[r]?.cells[c + 1]
                if (cell) cell.classList.add('entry-cell-selected')
            }
        }
    }

    /** Resolves a click target to grid coordinates, or null if outside data cells. */
    function cellCoordsFromEvent(e: MouseEvent): { row: number; col: number } | null {
        const td = (e.target as HTMLElement).closest('td') as HTMLTableCellElement | null
        if (!td || !tbody.contains(td)) return null
        const tr = td.parentElement as HTMLTableRowElement
        const col = td.cellIndex - 1   // subtract pick label column
        const row = tr.rowIndex - 1     // subtract thead row
        if (row < 0 || col < 0 || row >= nPicks || col >= nDrafters) return null
        return { row, col }
    }

    let dragging = false

    table.addEventListener('mousedown', (e: MouseEvent) => {
        const coords = cellCoordsFromEvent(e)
        if (!coords) return
        if (e.shiftKey) {
            e.preventDefault()
            focusRow = coords.row
            focusCol = coords.col
        } else {
            anchorRow = coords.row
            anchorCol = coords.col
            focusRow  = coords.row
            focusCol  = coords.col
            dragging  = true
        }
        applySelectionHighlight()
    }, listenerOpts)

    table.addEventListener('mousemove', (e: MouseEvent) => {
        if (!dragging) return
        const coords = cellCoordsFromEvent(e)
        if (!coords) return
        if (coords.row !== focusRow || coords.col !== focusCol) {
            focusRow = coords.row
            focusCol = coords.col
            applySelectionHighlight()
        }
    }, listenerOpts)

    document.addEventListener('mouseup', () => { dragging = false }, listenerOpts)

    table.addEventListener('keydown', (e: KeyboardEvent) => {
        const ctrl = e.ctrlKey || e.metaKey
        if (!ctrl) return

        if (e.key === 'a') {
            // Select all data cells
            e.preventDefault()
            anchorRow = 0; anchorCol = 0
            focusRow = nPicks - 1; focusCol = nDrafters - 1
            applySelectionHighlight()
            return
        }

        if (e.key === 'c') {
            // Copy writes registry names (tab-separated), so the clipboard stays human-readable
            // and round-trips through the name-resolving paste below.
            e.preventDefault()
            const { r1, r2, c1, c2 } = selectionBounds()
            const lines: string[] = []
            for (let r = r1; r <= r2; r++) {
                const cols: string[] = []
                for (let c = c1; c <= c2; c++) {
                    const value = selects[r]?.[c]?.getValue() ?? ''
                    cols.push(value ? getRegistryEntry(Number(value)).name : '')
                }
                lines.push(cols.join('\t'))
            }
            navigator.clipboard.writeText(lines.join('\n'))

        } else if (e.key === 'v') {
            e.preventDefault()
            const { r1, c1 } = selectionBounds()

            navigator.clipboard.readText().then(text => {
                if (!text) return
                const lines = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n').trimEnd().split('\n')

                let changed = false
                const unmatchedNames: string[] = []
                for (let dr = 0; dr < lines.length; dr++) {
                    const r = r1 + dr
                    if (r >= nPicks) break
                    const values = lines[dr].split('\t')
                    for (let dc = 0; dc < values.length; dc++) {
                        const d = c1 + dc
                        if (d >= nDrafters) break
                        if (!selects[r]?.[d]) continue
                        const pastedName = values[dc].trim()
                        const pastedPlayerId = pastedName ? resolveRosterEntryName(pastedName) : null
                        // A name the registry doesn't know leaves the cell blank and is
                        // reported below — never silently kept or guessed at.
                        if (pastedName && pastedPlayerId === null) unmatchedNames.push(pastedName)
                        selects[r][d].setValue(pastedPlayerId === null ? '' : String(pastedPlayerId))
                        changed = true
                    }
                }

                pasteWarning.hidden = unmatchedNames.length === 0
                pasteWarning.textContent = unmatchedNames.length > 0
                    ? `Not recognized (left blank): ${unmatchedNames.join(', ')}`
                    : ''
                if (changed) rebuildInspector()
            })
        }
    }, listenerOpts)

    scroll.append(table)
    leftEl.append(scroll)

    // Paste feedback: names the registry doesn't recognize are listed here so a partial
    // paste can't silently masquerade as a complete one.
    const pasteWarning = document.createElement('div')
    pasteWarning.className = 'paste-warning'
    pasteWarning.hidden = true
    leftEl.append(pasteWarning)

    // ── Right: team selector + G-score table ─────────────────────────────────

    const wrap = document.createElement('div')
    wrap.className = 'seat-selector-wrap'

    const label = document.createElement('div')
    label.className   = 'pick-control-label'
    label.textContent = 'Which team do you want to inspect?'
    wrap.append(label)

    const teamSel = makeCustomSelect(
        'sr-team-select',
        teamNames.map((name, index) => ({ value: name, label: isLivePlatform ? name : getTeamLabel(index) })),
        undefined,
        undefined,
        rosterListenerController.signal,
    )
    wrap.append(teamSel.element)
    rightEl.append(wrap)

    // Container for the G-score table — rebuilt when team or roster changes
    const tableContainer = document.createElement('div')
    rightEl.append(tableContainer)

    function rebuildInspector(): void {
        const selectedTeam = teamSel.getValue()
        const teamIdx = teamNames.indexOf(selectedTeam)
        if (teamIdx < 0) { tableContainer.innerHTML = ''; return }
        if (isPointsMode()) {
            buildTeamPointsTable(teamIdx, selects, nPicks, teamNames, tableContainer)
        } else {
            buildTeamGScoreTable(teamIdx, selects, nPicks, teamNames, tableContainer)
        }
    }

    // Rebuild when the team selector changes
    teamSel.element.addEventListener('change', rebuildInspector, listenerOpts)

    // Rebuild when any roster select changes
    for (const rowSelects of selects) {
        for (const sel of rowSelects) {
            sel.element.addEventListener('change', rebuildInspector, listenerOpts)
        }
    }

    // Initial render
    rebuildInspector()
}

// ─── G-score team inspector table ────────────────────────────────────────────

/**
 * Builds a G-score table for the selected team: one row per rostered player
 * plus a totals row, followed by an H-score row fetched from the backend.
 * Styled to match the expanded-view G-score tables.
 */
async function buildTeamGScoreTable(
    teamIdx: number
    , selects: CustomSelect[][]
    , nPicks: number
    , teamNames: string[]
    , container: HTMLElement
): Promise<void> {
    container.innerHTML = ''
    const categories = getSelectedCategories()
    const gScoreMap  = getGScoreById()

    // Collect G-scores for players on this team
    const rows: { playerId: number; values: number[]; total: number }[] = []
    for (let r = 0; r < nPicks; r++) {
        const playerId = readSelectedPlayerId(selects[r][teamIdx])
        if (playerId === null) continue
        const gs = gScoreMap.get(playerId)
        if (!gs) throw new Error(`G-score not found for player id ${playerId}`)
        rows.push({ playerId, values: gs.values, total: gs.total })
    }

    if (rows.length === 0) return

    // ── Build G-score table ──────────────────────────────────────────────────

    const tbl = document.createElement('table')
    tbl.className = 'panel-table'
    tbl.style.tableLayout = 'fixed'
    tbl.dataset.testid = 'roster-inspection-gscore'

    // Spacer row to lock column widths
    const tHead = tbl.createTHead()
    const spacerRow = tHead.insertRow(-1)
    spacerRow.style.border = 'none'
    spacerRow.appendChild(makeSpacerTh('panel-colspacer-name'))
    spacerRow.appendChild(makeSpacerTh('panel-colspacer-total'))
    for (let i = 0; i < categories.length; i++) spacerRow.appendChild(makeSpacerTh())

    // Header row — on mobile, swap full category names for their short form
    // (e.g. "Points" → "Pts") to keep columns narrow.
    const isMobile   = isMobileViewport()
    const shortNames = isMobile ? getShortCategoryNames() : {}
    const headerRow = tHead.insertRow(-1)
    headerRow.appendChild(makeSpacerTh())  // invisible label spacer
    const totalTh = document.createElement('th')
    totalTh.className = 'panel-colheader'
    totalTh.textContent = 'Total'
    headerRow.appendChild(totalTh)
    for (const cat of categories) {
        const label = shortNames[cat] ?? cat
        const th = document.createElement('th')
        th.className = label.length >= 10 ? 'panel-colheader colheader-long' : 'panel-colheader'
        th.textContent = label
        headerRow.appendChild(th)
    }

    // Data rows — one per player
    const tBody = tbl.createTBody()
    const teamTotals = new Array(categories.length).fill(0)
    let teamTotalSum = 0

    for (const row of rows) {
        const tr = tBody.insertRow(-1)

        const labelCell = document.createElement('th')
        labelCell.className = 'panel-rowlabel'
        labelCell.append(makeFullPlayerDisplay(row.playerId))
        tr.appendChild(labelCell)

        const totalCell = tr.insertCell(-1)
        totalCell.textContent = row.total.toFixed(2)
        totalCell.className = 'panel-datacell celltypea'
        teamTotalSum += row.total

        for (let i = 0; i < row.values.length; i++) {
            const cell = tr.insertCell(-1)
            cell.textContent = row.values[i].toFixed(2)
            cell.style.cssText = stat_styler_primary(row.values[i], 60, 0)
            cell.className = 'panel-datacell'
            teamTotals[i] += row.values[i]
        }
    }

    // Totals row
    const totalsRow = tBody.insertRow(-1)
    const totalsLabel = document.createElement('th')
    totalsLabel.className = 'panel-rowlabel'
    totalsLabel.textContent = 'Team Total'
    totalsRow.appendChild(totalsLabel)

    const totalsCell = totalsRow.insertCell(-1)
    totalsCell.textContent = teamTotalSum.toFixed(2)
    totalsCell.className = 'panel-datacell celltypeb'

    for (const val of teamTotals) {
        const cell = totalsRow.insertCell(-1)
        cell.textContent = val.toFixed(2)
        cell.style.cssText = stat_styler_primary(val, 60, 0)
        cell.className = 'panel-datacell'
    }

    container.appendChild(tbl)

    // ── H-score row (fetched from backend, only for full rosters) ────────────

    if (rows.length < nPicks) return

    const nDrafters = teamNames.length
    const playerAssignments: Record<string, number[]> = {}
    for (let d = 0; d < nDrafters; d++) {
        const team = teamNames[d]
        const players: number[] = []
        for (let r = 0; r < nPicks; r++) {
            const playerId = readSelectedPlayerId(selects[r][d])
            if (playerId !== null) players.push(playerId)
        }
        playerAssignments[team] = players
    }

    const teamName = teamNames[teamIdx]
    const result = await evaluateTeamHScore(playerAssignments, teamName)
    if (!result) return

    const isRoto     = getScoringFormat() === 'Rotisserie'
    const rotoNDrafters = isRoto ? getLeagueSettings().n_drafters : 0
    const rotoMiddle = (rotoNDrafters - 1) / 2 + 1

    const hScoreTbl = document.createElement('table')
    hScoreTbl.className = 'panel-table panel-table--rounded panel-table--top-gap'
    hScoreTbl.style.tableLayout = 'fixed'
    hScoreTbl.dataset.testid = 'roster-inspection-hscore'

    // Match the team-inspector's column widths above so the two tables line up.
    // Mobile values mirror the #rosters-right .panel-colspacer-* overrides.
    const colgroup = document.createElement('colgroup')
    const nameCol  = document.createElement('col')
    nameCol.style.width = isMobile ? '7rem' : '200px'
    const totalCol = document.createElement('col')
    totalCol.style.width = isMobile ? '3rem' : '83px'
    colgroup.append(nameCol, totalCol)
    for (let i = 0; i < categories.length; i++) colgroup.appendChild(document.createElement('col'))
    hScoreTbl.appendChild(colgroup)

    const hScoreTBody = hScoreTbl.createTBody()
    const hScoreRow = hScoreTBody.insertRow(-1)

    const hScoreLabel = document.createElement('th')
    hScoreLabel.className = 'panel-rowlabel'
    hScoreLabel.textContent = 'H-Score (est. win rate)'
    hScoreRow.appendChild(hScoreLabel)

    const hScoreTotalCell = hScoreRow.insertCell(-1)
    hScoreTotalCell.textContent = result.h_score.toFixed(1)
    hScoreTotalCell.className = 'overallhscore'

    for (const winRate of result.win_rates) {
        const cell = hScoreRow.insertCell(-1)
        if (isRoto) {
            const rotoValue = 1 + (winRate / 100) * (rotoNDrafters - 1)
            cell.textContent = rotoValue.toFixed(1)
            cell.className = 'categoricalRotoHscore'
            cell.style.cssText = stat_styler_primary(rotoValue, 3 * (rotoNDrafters - 1), rotoMiddle)
        } else {
            cell.textContent = winRate.toFixed(1)
            cell.className = 'categoricalhscore'
            cell.style.cssText = stat_styler_primary(winRate, 3, 50)
        }
    }

    container.appendChild(hScoreTbl)
}

/**
 * Builds a Points Inspector table for the selected team when in Points mode:
 * Displays Player name, position, FP/G, total projected FP, and slot status
 * (Active starter slot vs Bench).
 */
function buildTeamPointsTable(
    teamIdx: number
    , selects: CustomSelect[][]
    , nPicks: number
    , teamNames: string[]
    , container: HTMLElement
): void {
    container.innerHTML = ''

    // Collect players on this team
    const teamPlayerIds: number[] = []
    for (let r = 0; r < nPicks; r++) {
        const playerId = readSelectedPlayerId(selects[r][teamIdx])
        if (playerId !== null) teamPlayerIds.push(playerId)
    }

    if (teamPlayerIds.length === 0) return

    let resultsMap: Map<number, PlayerResult> | null = null
    try {
        resultsMap = getPlayerResultsById()
    } catch {
        resultsMap = null
    }

    interface PlayerPointsRow {
        playerId: number
        name: string
        positions: string[]
        fpg: number
        totalFp: number
        slotStatus: string
        isStarter: boolean
    }

    // Retrieve slot configuration
    let slotCounts: Record<string, number> = {}
    try {
        slotCounts = getSlotCounts()
    } catch {
        slotCounts = { PG: 1, SG: 1, SF: 1, PF: 1, C: 1, G: 1, F: 1, UTIL: 3 }
    }

    // Build ordered list of active slots
    const baseSlots: string[] = []
    const flexSlots: string[] = []
    const utilSlots: string[] = []

    for (const [slot, count] of Object.entries(slotCounts)) {
        if (slot.toLowerCase() === 'bench' || slot.toLowerCase() === 'ir') continue
        for (let i = 0; i < count; i++) {
            const s = slot.toUpperCase()
            if (['PG', 'SG', 'SF', 'PF', 'C'].includes(s)) {
                baseSlots.push(s)
            } else if (['G', 'F'].includes(s)) {
                flexSlots.push(s)
            } else {
                utilSlots.push(s)
            }
        }
    }

    const activeSlotPool: string[] = [...baseSlots, ...flexSlots, ...utilSlots]
    if (activeSlotPool.length === 0) {
        activeSlotPool.push('PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL', 'UTIL', 'UTIL')
    }

    // Sort team players by FP/G descending for starter assignment
    const candidateRows = teamPlayerIds.map(pid => {
        const entry = getRegistryEntry(pid)
        const pr = resultsMap?.get(pid)
        const fpg = pr?.points_per_game ?? (pr?.total_projected_points ? pr.total_projected_points / 3.5 : 0)
        const totalFp = pr?.total_projected_points ?? (fpg * 3.5)
        return {
            playerId: pid,
            name: entry.name,
            positions: entry.positions,
            fpg,
            totalFp,
        }
    })

    candidateRows.sort((a, b) => b.fpg - a.fpg)

    // Assign slots to players
    const remainingSlots = [...activeSlotPool]
    const assignedRows: PlayerPointsRow[] = []

    for (const cand of candidateRows) {
        let assignedSlotIndex = -1
        // 1. Try base position match
        for (let i = 0; i < remainingSlots.length; i++) {
            const slot = remainingSlots[i]
            if (cand.positions.includes(slot)) {
                assignedSlotIndex = i
                break
            }
        }
        // 2. Try flex position match (G or F)
        if (assignedSlotIndex === -1) {
            for (let i = 0; i < remainingSlots.length; i++) {
                const slot = remainingSlots[i]
                if (slot === 'G' && (cand.positions.includes('PG') || cand.positions.includes('SG'))) {
                    assignedSlotIndex = i
                    break
                }
                if (slot === 'F' && (cand.positions.includes('SF') || cand.positions.includes('PF'))) {
                    assignedSlotIndex = i
                    break
                }
            }
        }
        // 3. Try UTIL match
        if (assignedSlotIndex === -1) {
            for (let i = 0; i < remainingSlots.length; i++) {
                if (remainingSlots[i] === 'UTIL') {
                    assignedSlotIndex = i
                    break
                }
            }
        }

        if (assignedSlotIndex !== -1) {
            const slot = remainingSlots.splice(assignedSlotIndex, 1)[0]
            assignedRows.push({
                ...cand,
                slotStatus: `Active (${slot})`,
                isStarter: true,
            })
        } else {
            assignedRows.push({
                ...cand,
                slotStatus: 'Bench',
                isStarter: false,
            })
        }
    }

    // Build Points Inspector DOM Table
    const tbl = document.createElement('table')
    tbl.className = 'panel-table panel-table--rounded'
    tbl.style.tableLayout = 'fixed'
    tbl.dataset.testid = 'roster-inspection-points'

    const isMobile = isMobileViewport()
    const thead = tbl.createTHead()

    // Spacer row
    const spacerRow = thead.insertRow(-1)
    spacerRow.style.border = 'none'
    const spName = makeSpacerTh('panel-colspacer-name')
    spName.style.width = isMobile ? '7rem' : '180px'
    const spPos = makeSpacerTh()
    spPos.style.width = isMobile ? '3rem' : '75px'
    const spFpg = makeSpacerTh()
    spFpg.style.width = isMobile ? '3rem' : '75px'
    const spTot = makeSpacerTh()
    spTot.style.width = isMobile ? '3.5rem' : '85px'
    const spSlot = makeSpacerTh()
    spSlot.style.width = isMobile ? '5.5rem' : '120px'
    spacerRow.append(spName, spPos, spFpg, spTot, spSlot)

    // Header row
    const headerRow = thead.insertRow(-1)
    const thPlayer = document.createElement('th')
    thPlayer.className = 'panel-colheader'
    thPlayer.textContent = 'Player'
    thPlayer.style.textAlign = 'left'

    const thPos = document.createElement('th')
    thPos.className = 'panel-colheader'
    thPos.textContent = 'Pos'

    const thFpg = document.createElement('th')
    thFpg.className = 'panel-colheader'
    thFpg.textContent = 'FP/G'

    const thTot = document.createElement('th')
    thTot.className = 'panel-colheader'
    thTot.textContent = 'Proj FP'

    const thSlot = document.createElement('th')
    thSlot.className = 'panel-colheader'
    thSlot.textContent = 'Slot / Status'

    headerRow.append(thPlayer, thPos, thFpg, thTot, thSlot)

    const tbody = tbl.createTBody()
    let activeFpSum = 0
    let benchFpSum = 0
    let activeCount = 0

    for (const row of assignedRows) {
        const tr = tbody.insertRow(-1)

        const nameCell = document.createElement('th')
        nameCell.className = 'panel-rowlabel'
        nameCell.append(makeFullPlayerDisplay(row.playerId))
        tr.appendChild(nameCell)

        const posCell = tr.insertCell(-1)
        posCell.className = 'panel-datacell'
        posCell.textContent = row.positions.join('/')

        const fpgCell = tr.insertCell(-1)
        fpgCell.className = 'panel-datacell'
        fpgCell.textContent = row.fpg > 0 ? row.fpg.toFixed(1) : '—'

        const totCell = tr.insertCell(-1)
        totCell.className = 'panel-datacell celltypea'
        totCell.textContent = row.totalFp > 0 ? row.totalFp.toFixed(1) : '—'

        const slotCell = tr.insertCell(-1)
        slotCell.className = 'panel-datacell'
        const badge = document.createElement('span')
        if (row.isStarter) {
            badge.textContent = row.slotStatus
            badge.style.cssText = 'display:inline-block; padding: 2px 7px; border-radius: 4px; font-weight: 600; font-size: 0.72rem; background: rgba(56, 161, 105, 0.2); color: #48bb78; border: 1px solid rgba(72, 187, 120, 0.4);'
            activeFpSum += row.totalFp
            activeCount++
        } else {
            badge.textContent = 'Bench'
            badge.style.cssText = 'display:inline-block; padding: 2px 7px; border-radius: 4px; font-weight: 500; font-size: 0.72rem; background: rgba(160, 174, 192, 0.15); color: #a0aec0; border: 1px solid rgba(160, 174, 192, 0.3);'
            benchFpSum += row.totalFp
        }
        slotCell.append(badge)
    }

    // Totals rows
    // Active starters total
    const activeRow = tbody.insertRow(-1)
    const activeLabel = document.createElement('th')
    activeLabel.className = 'panel-rowlabel'
    activeLabel.textContent = 'Active Starters Total'
    activeRow.appendChild(activeLabel)
    activeRow.insertCell(-1).textContent = ''
    activeRow.insertCell(-1).textContent = ''
    const activeTotalCell = activeRow.insertCell(-1)
    activeTotalCell.className = 'panel-datacell celltypeb'
    activeTotalCell.textContent = activeFpSum.toFixed(1)
    const activeStatusCell = activeRow.insertCell(-1)
    activeStatusCell.className = 'panel-datacell'
    activeStatusCell.textContent = `${activeCount} Starters`

    // Bench total
    const benchRow = tbody.insertRow(-1)
    const benchLabel = document.createElement('th')
    benchLabel.className = 'panel-rowlabel'
    benchLabel.textContent = 'Bench Total'
    benchRow.appendChild(benchLabel)
    benchRow.insertCell(-1).textContent = ''
    benchRow.insertCell(-1).textContent = ''
    const benchTotalCell = benchRow.insertCell(-1)
    benchTotalCell.className = 'panel-datacell'
    benchTotalCell.textContent = benchFpSum.toFixed(1)
    const benchStatusCell = benchRow.insertCell(-1)
    benchStatusCell.className = 'panel-datacell'
    benchStatusCell.textContent = `${assignedRows.length - activeCount} Bench`

    // Full roster total
    const totalRow = tbody.insertRow(-1)
    const totalLabel = document.createElement('th')
    totalLabel.className = 'panel-rowlabel'
    totalLabel.textContent = 'Team Projected FP'
    totalRow.appendChild(totalLabel)
    totalRow.insertCell(-1).textContent = ''
    totalRow.insertCell(-1).textContent = ''
    const totalCell = totalRow.insertCell(-1)
    totalCell.className = 'panel-datacell celltypeb'
    totalCell.textContent = (activeFpSum + benchFpSum).toFixed(1)
    totalRow.insertCell(-1).textContent = ''

    container.appendChild(tbl)

    // Matchup Points Summary Card
    const summaryCard = document.createElement('div')
    summaryCard.className = 'panel-table panel-table--rounded panel-table--top-gap'
    summaryCard.style.cssText = 'padding: 12px 16px; background: light-dark(#f8fafc, #161a23); border: 1px solid light-dark(#e2e8f0, #2b3040); border-radius: 6px; display: flex; justify-content: space-between; align-items: center;'
    summaryCard.innerHTML = `
        <div>
            <div style="font-size: 0.72rem; color: #889; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;">Weekly Starting Lineup Projected Points</div>
            <div style="font-size: 1.35rem; font-weight: 700; color: #48bb78;">${activeFpSum.toFixed(1)} <span style="font-size: 0.8rem; font-weight: normal; color: #889;">FP</span></div>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 0.72rem; color: #889; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;">Active Lineup Slots</div>
            <div style="font-size: 0.95rem; font-weight: 600; color: light-dark(#333, #dde);">${activeCount} of ${assignedRows.length} Rostered</div>
        </div>
    `
    container.appendChild(summaryCard)
}

/** Creates an invisible `<th>` spacer to lock column widths in panel tables. */
function makeSpacerTh(extraClass?: string): HTMLTableCellElement {
    const th = document.createElement('th')
    th.className = extraClass ? `panel-colspacer ${extraClass}` : 'panel-colspacer'
    return th
}

/** Resolves a human-entered roster name to a player id, or null when the registry doesn't
 *  know it. Accepts both bare names and the pre-refactor "Name (POS)" form, which
 *  DEFAULT_SEASON_ROSTERS (and old clipboard copies) still carry. */
function resolveRosterEntryName(entryName: string): number | null {
    const bareName = entryName.replace(/\s*\([A-Z,\-]+\)\s*$/, '')
    return findPlayerIdByName(bareName)
}

/** The player id a roster select holds, or null when the cell is blank. Throws on a
 *  non-numeric value — the selects only ever carry stringified ids. */
function readSelectedPlayerId(select: CustomSelect): number | null {
    const value = select.getValue()
    if (!value) return null
    const playerId = Number(value)
    if (Number.isNaN(playerId)) throw new Error(`Roster select carried a non-numeric value: "${value}"`)
    return playerId
}
