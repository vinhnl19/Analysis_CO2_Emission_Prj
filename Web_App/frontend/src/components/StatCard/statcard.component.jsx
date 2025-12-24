import Card from "antd/es/card/Card"
import styles from "./statcard.component.module.css"
import { Flex, Space, Typography, Skeleton } from "antd"
import { InfoCircleOutlined } from '@ant-design/icons'
import { indicatorIconMap } from "../../constants/indicatorIconMap"
import { formatNumber } from "../../utils/formatNumber"

const { Text } = Typography
export default function StatCardComponent(
    {
        icon,
        value,
        unitName,
        description,
        loading = false
    }) {
    const defaultIcon = <InfoCircleOutlined style={{ fontSize: '24px', color: '#1890ff' }} />;
    return (
        <Card>
            <Flex vertical gap={0}>
                <div className={styles.icon}>
                    {loading ? (
                        <Skeleton.Avatar active size={32} shape="circle" />
                    ) : (
                        indicatorIconMap[icon] || defaultIcon
                    )}
                </div>
                <Space style={{ marginTop: '12px' }}>
                    {loading ? (
                        <Skeleton.Input active size="small" style={{ width: 80, height: '20px' }} />
                    ) : (
                        <>
                            <Text className={styles.valueCard} strong>
                                {formatNumber(value)}
                            </Text>
                            <Text className={styles.unitCard} strong>
                                {unitName}
                            </Text>
                        </>
                    )}
                </Space>
                {loading ? (
                    <div className={styles.skeletonDescription}>
                        <Skeleton.Input active size="small" style={{ width: 220, height: '16px' }} />
                    </div>
                ) : (
                    <Text className={styles.descriptionCard}>
                        {description}
                    </Text>
                )}
            </Flex>
        </Card>
    )
}
