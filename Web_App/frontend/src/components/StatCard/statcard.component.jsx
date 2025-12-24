import Card from "antd/es/card/Card"
import styles from "./statcard.component.module.css"
import { Flex, Space, Typography, Skeleton, Popover } from "antd"
import { InfoCircleOutlined, InfoCircleTwoTone } from '@ant-design/icons'
import { indicatorIconMap } from "../../constants/indicatorIconMap"
import { formatNumber } from "../../utils/formatNumber"

const { Text } = Typography
export default function StatCardComponent(
    {
        icon,
        value,
        unitName,
        description,
        calculationNote,
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
                    <Space>
                        <Text className={styles.descriptionCard}>
                            {description}
                        </Text>
                        {calculationNote && (
                            <Popover
                                title="Calculation Details"
                                content={
                                    <div className={styles.popoverContent}>
                                        {calculationNote}
                                    </div>
                                }
                                placement="top"
                                trigger="hover"
                            >
                                <InfoCircleTwoTone
                                    style={{ fontSize: 12, cursor: 'pointer' }}
                                />
                            </Popover>

                        )}
                    </Space>
                )}
            </Flex>
        </Card>
    )
}
