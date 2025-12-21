import {createBrowserRouter} from 'react-router-dom'
import AppLayout from './layout/AppLayout'
import DashboardPage from './pages/Dashboard/Dashboard.page'
import ForcastingPage from './pages/Forcasting/forcasting.page'
import RecommendationPage from './pages/Recommendation/recommendation.page'

export const router = createBrowserRouter([
    {
        path: "/",
        element: <AppLayout/>,
        children: [
            {path: "/", element: <DashboardPage/>},
            {path: "/forcasting", element: <ForcastingPage/>},
            {path: "/recommendation", element: <RecommendationPage/>}
        ]
    }
])