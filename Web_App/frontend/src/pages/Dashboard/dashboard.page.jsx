import { Select, Form, Slider, Typography, Row, Col, Flex } from 'antd'
import styles from "./dashboard.page.module.css"
import StatCardComponent from '../../components/StatCard/statcard.component'
import { GlobalOutlined } from '@ant-design/icons'

const { Text } = Typography

export default function DashboardPage() {
  // fixed data
  const fixedCountrys = [
    {
      value: 'vn', label: 'Viet Nam'
    },
    {
      value: 'us', label: 'United States'
    }
  ]
  const fixedContinent = [
    {
      value: 'asia', label: 'Asia'
    },
    {
      value: 'euroupe', label: 'Euroupe'
    }
  ]
  const fixedMinYear = 2001
  const fixedMaxYear = 2022
  const fixedMarksYear = {
    [fixedMinYear]: <span className={styles.sliderMark}>{fixedMinYear}</span>,
    [fixedMaxYear]: <span className={styles.sliderMark}>{fixedMaxYear}</span>
  }
  const dataStatCard = [
    {
      icon: <GlobalOutlined style={{ fontSize: 24, color: '#0D59F2' }} />,
      value: "5200",
      unitName: "MtCO2",
      description: "Total CO₂ Emissions"
    },
    {
      icon: <GlobalOutlined style={{ fontSize: 24, color: '#0D59F2' }} />,
      value: "15.7",
      unitName: "tCO2",
      description: "CO₂ per Capita"
    },
    {
      icon: <GlobalOutlined style={{ fontSize: 24, color: '#0D59F2' }} />,
      value: "26.9",
      unitName: "Tỉ",
      description: "Population"
    },
        {
      icon: <GlobalOutlined style={{ fontSize: 24, color: '#0D59F2' }} />,
      value: "0.93",
      unitName: "$",
      description: "GDP"
    },
        {
      icon: <GlobalOutlined style={{ fontSize: 24, color: '#0D59F2' }} />,
      value: "0.93",
      unitName: "$",
      description: "Government Expenditure on Education"
    },    {
      icon: <GlobalOutlined style={{ fontSize: 24, color: '#0D59F2' }} />,
      value: "5200",
      unitName: "MtCO2",
      description: "Total Energy "
    },    {
      icon: <GlobalOutlined style={{ fontSize: 24, color: '#0D59F2' }} />,
      value: "5200",
      unitName: "MWh",
      description: "Total CO₂ Emissions"
    },    {
      icon: <GlobalOutlined style={{ fontSize: 24, color: '#0D59F2' }} />,
      value: "15.7",
      unitName: "",
      description: "CRI"
    },    {
      icon: <GlobalOutlined style={{ fontSize: 24, color: '#0D59F2' }} />,
      value: "26.9",
      unitName: "Ha",
      description: "Area"
    }
  ]
  // filter option
  const filterOptionFunc = (input, option) => {
    return (option?.label ?? '').toLowerCase().includes(input.toLowerCase())
  }
  return (
    <div>
      <Form layout='vertical'>
        {/* Label */}
        <Row gutter={24} style={{marginBottom: '6px'}}>
          <Col span={6}>
            <Text className={styles.labelInputFilter} strong>Continent</Text>
          </Col>
          <Col span={6}>
            <Text className={styles.labelInputFilter} strong>Country</Text>
          </Col>
          <Col span={10}>
            <Text className={styles.labelInputFilter} strong>Period</Text>
          </Col>
        </Row>
        {/* form item */}
        <Row gutter={24}>
          <Col span={6}>
              <Form.Item name='continent'>
                <Select
                  mode='multiple'
                  allowClear
                  showSearch={{ filterOption: filterOptionFunc }}
                  placeholder="Select a continent"
                  options={fixedContinent}
                ></Select>
              </Form.Item>
          </Col>
          <Col span={6}>
              <Form.Item name='country'>
                <Select
                  mode='multiple'
                  allowClear
                  showSearch={{ filterOption: filterOptionFunc }}
                  placeholder="Select a country"
                  options={fixedCountrys}
                ></Select>
              </Form.Item>
          </Col>
          <Col span={8}>
              <Form.Item name="period">
                <Slider
                  range
                  min={fixedMinYear}
                  max={fixedMaxYear}
                  step={1}
                  marks={fixedMarksYear}
                />
              </Form.Item>
          </Col>
        </Row>
      </Form>
      <Flex wrap gap="small" align='middle' justify='center'>
        {dataStatCard.map((item, index) => (
          <div key={index} className={styles.statCardWrapper}>
            <StatCardComponent
              key={index}
              icon={item.icon}
              value={item.value}
              unitName={item.unitName}
              description={item.description}
            />
          </div>
        ))}
      </Flex>
    </div>
  )
}
