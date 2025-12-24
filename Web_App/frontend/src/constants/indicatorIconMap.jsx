import {
    GlobalOutlined,
    CloudOutlined,
    UserOutlined,
    DollarOutlined,
    BookOutlined,
    ThunderboltOutlined,
    WarningOutlined,
    ExpandOutlined,
    BarChartOutlined
} from "@ant-design/icons";

export const indicatorIconMap = {
    co2global: <GlobalOutlined className="statcard-icon" />,
    co2capita: <CloudOutlined className="statcard-icon" />,
    population: <UserOutlined className="statcard-icon" />,
    gdp: <DollarOutlined className="statcard-icon" />,
    education: <BookOutlined className="statcard-icon" />,
    energy: <ThunderboltOutlined className="statcard-icon" />,
    cri: <WarningOutlined className="statcard-icon" />,
    area: <ExpandOutlined className="statcard-icon" />,
    hdi: <BarChartOutlined className="statcard-icon" />
};