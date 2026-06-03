import { useState } from 'react'
import Layout from './components/Layout'
import Overview from './pages/Overview'
import PlayerRatings from './pages/PlayerRatings'
import PlayerStyles from './pages/PlayerStyles'
import TeamStyles from './pages/TeamStyles'
import StyleMatch from './pages/StyleMatch'
import ClubAudit from './pages/ClubAudit'
import ModelLab from './pages/ModelLab'

const PAGES: Record<string, () => JSX.Element> = {
  overview: Overview,
  ratings: PlayerRatings,
  'player-styles': PlayerStyles,
  'team-styles': TeamStyles,
  match: StyleMatch,
  audit: ClubAudit,
  lab: ModelLab,
}

export default function App() {
  const [page, setPage] = useState('overview')
  const Page = PAGES[page] || Overview

  return (
    <Layout activePage={page} onNavigate={setPage}>
      <Page key={page} />
    </Layout>
  )
}
