import Card from "antd/es/card/Card"
import styles from "./statcard.component.module.css"
import { Flex, Space, Typography } from "antd"
import { InfoCircleOutlined } from '@ant-design/icons'

const { Text } = Typography
export default function StatCardComponent(
    { 
        icon, 
        value, 
        unitName, 
        description 
    }) {
    const defaultIcon = <InfoCircleOutlined style={{ fontSize: '24px', color: '#1890ff' }} />;
    return (
        <Card>
            <Flex vertical gap={0}>
                <div className={styles.icon}>
                    {icon || defaultIcon}
                </div>
                <Space style={{marginTop: '12px'}}>
                    <Text className={styles.valueCard} strong>{value}</Text>
                    <Text className={styles.unitCard} strong>{unitName}</Text>
                </Space>
                <Text className={styles.descriptionCard}>{description}</Text>
            </Flex>
        </Card>
    )
}
