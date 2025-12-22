import { Select, Form, Slider, Typography, Row, Col, Flex } from 'antd'
import styleDashboard from "./dashboard.page.module.css"
import '../../styles/typography.css'
import StatCardComponent from '../../components/StatCard/statcard.component'
import { GlobalOutlined } from '@ant-design/icons'
import { useCountries } from '../../hooks/useCountries'
import { useEffect } from 'react'
import { useIndicatorCards } from '../../hooks/useIndicators'

const { Text } = Typography

export default function DashboardPage() {
  const { data: dataCoCo, isLoading: isLoadingCoCo } = useCountries()
  const [form] = Form.useForm()
  const selectedContinents = Form.useWatch('continent', form)
  const selectedCountries = Form.useWatch('country', form)
  const continentOptions = Array.from(
    new Set(dataCoCo?.map(item => item.continent))
  ).map(cont => ({
    label: cont,
    value: cont
  }))
  const countryOptions = dataCoCo?.filter(item => {
    if (!selectedContinents || selectedContinents.length === 0) return true
    return selectedContinents.includes(item.continent)
  }).map(item => ({
    label: item.country_name,
    value: item.country_code
  }))
  useEffect(() => {
    if (!selectedContinents || selectedContinents.length === 0) {
      // Không chọn continent → xoá hết country
      form.setFieldsValue({ country: [] })
      return
    }

    if (!selectedCountries || selectedCountries.length === 0) {
      return
    }

    // Map country_code -> continent
    const countryToContinentMap = new Map(
      dataCoCo.map(item => [item.country_code, item.continent])
    )

    // Giữ lại country còn thuộc continent mới
    const validCountries = selectedCountries.filter(code =>
      selectedContinents.includes(countryToContinentMap.get(code))
    )

    // Nếu danh sách thay đổi thì mới set lại (tránh re-render dư)
    if (validCountries.length !== selectedCountries.length) {
      form.setFieldsValue({ country: validCountries })
    }

  }, [selectedContinents, selectedCountries, dataCoCo, form])

  //fixed payload
  const fixedPayload = {
    country_code_list: ["CHN"],
    fromYear: 2020,
    toYear: 2020
  }
  const { data: dataIndicators, isLoading: isLoadingIndicators } = useIndicatorCards(
    fixedPayload.country_code_list, fixedPayload.fromYear, fixedPayload.toYear
  )

  // fixed data
  const fixedMinYear = 2001
  const fixedMaxYear = 2022
  const fixedMarksYear = {
    [fixedMinYear]: <span className={styleDashboard.sliderMark}>{fixedMinYear}</span>,
    [fixedMaxYear]: <span className={styleDashboard.sliderMark}>{fixedMaxYear}</span>
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
    }, {
      icon: <GlobalOutlined style={{ fontSize: 24, color: '#0D59F2' }} />,
      value: "5200",
      unitName: "MtCO2",
      description: "Total Energy "
    }, {
      icon: <GlobalOutlined style={{ fontSize: 24, color: '#0D59F2' }} />,
      value: "5200",
      unitName: "MWh",
      description: "Total CO₂ Emissions"
    }, {
      icon: <GlobalOutlined style={{ fontSize: 24, color: '#0D59F2' }} />,
      value: "15.7",
      unitName: "",
      description: "CRI"
    }, {
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
      <Form layout='vertical' form={form}>
        {/* Label */}
        <Row gutter={24} style={{ marginBottom: '6px' }}>
          <Col span={6}>
            <Text className='label-input-filter' strong>Continent</Text>
          </Col>
          <Col span={6}>
            <Text className='label-input-filter' strong>Country</Text>
          </Col>
          <Col span={10}>
            <Text className='label-input-filter' strong>Period</Text>
          </Col>
        </Row>
        {/* form item */}
        <Row gutter={24}>
          <Col span={6}>
            <Form.Item name='continent'>
              <Select
                mode='multiple'
                isLoading={isLoadingCoCo}
                allowClear
                showSearch={{ filterOption: filterOptionFunc }}
                placeholder="Select continents"
                options={continentOptions}
              ></Select>
            </Form.Item>
          </Col>
          <Col span={6}>
            <Form.Item name='country'>
              <Select
                mode='multiple'
                allowClear
                disabled={!selectedContinents?.length}
                isLoading={isLoadingCoCo}
                showSearch={{ filterOption: filterOptionFunc }}
                placeholder={selectedContinents?.length > 0 ? "Select countries" : "Please select continent first"}
                options={countryOptions}
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
        {dataIndicators?.map((item, index) => (
          <div key={index} className={styleDashboard.statCardWrapper}>
            <StatCardComponent
              key={index}
              icon={item.icon}
              value={item.value}
              unitName={item.unit}
              description={item.description}
            />
          </div>
        ))}
      </Flex>
    </div>
  )
}
