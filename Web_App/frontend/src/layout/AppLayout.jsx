import { Divider, Layout  } from 'antd'
import { Outlet } from 'react-router-dom'
import HeaderComponent from '../components/Header/header.component'
import FooterComponent from '../components/Footer/footer.component'
import NavTabsComponent from '../components/NavTabs/navtabs.component'

export default function AppLayout() {

    return (
        <Layout style={{ minHeight: '100vh', minWidth: '100vw', background: '#fff'}}>
            <div className="app-header">
                <HeaderComponent></HeaderComponent>
            </div>

            <div className="app-function-tabs">
                <NavTabsComponent></NavTabsComponent>
            </div>
            <Divider className='app-divider'></Divider>

            <div className="app-content">
                <Outlet></Outlet>
            </div>

            <div className="app-footer">
                <FooterComponent></FooterComponent>
            </div>

        </Layout>
    )
}