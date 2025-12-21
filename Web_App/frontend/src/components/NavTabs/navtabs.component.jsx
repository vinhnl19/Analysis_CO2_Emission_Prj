import { Tabs } from "antd";
import { useLocation, useNavigate } from "react-router-dom";
import { PieChartTwoTone, FundTwoTone, SlidersTwoTone} from "@ant-design/icons"
import style from "./navtabs.component.module.css"

export default function NavTabsComponent() {
    const navigate = useNavigate()
    const location = useLocation()

    const items = [
        { key: "/", label: "Dashboard", icon: <PieChartTwoTone/>},
        { key: "/forcasting", label: "Forcast CO2 Emission", icon: <FundTwoTone />},
        { key: "/recommendation", label: "Recommendation Engine", icon: <SlidersTwoTone /> }
    ]

    return (
        <Tabs
            className={style.navbarTabs}
            activeKey={location.pathname}
            items={items}
            onChange={(key) => navigate(key)}
        ></Tabs>
    )
}