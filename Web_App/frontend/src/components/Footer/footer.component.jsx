import Typography from "antd/es/typography/Typography";
import style from "./footer.component.module.css"

const { Text } = Typography

export default function FooterComponent() {
    return (
        <Text className={style.footer}>
            Sản phẩm được thực hiện phục vụ nghiên cứu khoa học.
        </Text>
    )
}