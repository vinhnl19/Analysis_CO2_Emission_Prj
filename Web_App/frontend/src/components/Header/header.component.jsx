import Typography from "antd/es/typography/Typography"
import style from "./header.component.module.css"

const { Text } = Typography

export default function HeaderComponent() {
    return (
        <Text className={style.title}>CO2 Emission</Text>
    )
}