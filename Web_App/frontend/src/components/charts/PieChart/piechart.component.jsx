import {
    PieChart as RePieChart, 
    Pie,
    Cell,
    Tooltip,
    Legend,
    ResponsiveContainer
} from 'recharts'
import { formatNumber } from '../../../utils/formatNumber'
import { Flex, Skeleton } from 'antd';

const COLORS = ["#033e0c", "#668f6b", "#9fcab3", "#cfeedd"];

export default function PieChartComponent({
    data = [],
    height = 300,
    titleChart = "",
    loading = false
}) {
    const hasData = data?.some(d => typeof d.value === "number");

    if (loading) {
        return (
            <Flex orientation='vertical' align='center' gap={14} justify='center'>
                <Skeleton.Input
                    active
                    size="small"
                    style={{ width: 180, marginTop: "5px" }}
                />
                <Skeleton.Avatar
                    active
                    shape="circle"
                    size={200}
                />
                <Skeleton.Input
                    active
                    size="small"
                    style={{ width: 250 }}
                />
            </Flex>
        );
    }

    if (!hasData) {
        return (
            <div
                style={{
                    height,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    color: "#668f6b"
                }}
            >
                No data available
            </div>
        );
    }
    const renderInnerLabel = ({
        cx,
        cy,
        midAngle,
        innerRadius,
        outerRadius,
        percent,
    }) => {
        const RADIAN = Math.PI / 180;
        const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
        const x = cx + radius * Math.cos(-midAngle * RADIAN);
        const y = cy + radius * Math.sin(-midAngle * RADIAN);

        return (
            <text
                x={x}
                y={y}
                fill="#ffffff"
                textAnchor="middle"
                dominantBaseline="central"
                fontSize={12}
                fontWeight={500}
            >
                {`${(percent * 100).toFixed(0)}%`}
            </text>
        );
    };


    return (
        <ResponsiveContainer width="100%" height={height}>
            <RePieChart>
                <text
                    x="50%"
                    y="6%"
                    textAnchor="middle"
                    dominantBaseline="middle"
                    fontSize={16}
                    fontWeight={600}
                    fill="#033e0c"
                >
                    {titleChart}
                </text>
                <Pie 
                    data={data}
                    dataKey="value"
                    nameKey="label"
                    cy="53%"
                    outerRadius="80%"
                    paddingAngle={0}
                    label={renderInnerLabel}
                    labelLine={false}
                >
                    {data.map((_, index) => (
                        <Cell
                            key={index}
                            fill={COLORS[index % COLORS.length]}
                        >
                        </Cell>
                    ))}
                </Pie>
                <Legend verticalAlign="bottom" height={36} align='center'/>
                <Tooltip
                    formatter={(value, name, props) => 
                        `${formatNumber(value)} ${props.payload.unit ?? ""}`
                    }
                >
                </Tooltip>
            </RePieChart>
        </ResponsiveContainer>
    )
}