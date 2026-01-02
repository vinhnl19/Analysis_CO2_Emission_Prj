import { Select, Form, Slider, Typography, Row, Col, Flex, Space } from 'antd'
import styleDashboard from "./dashboard.page.module.css"
import '../../styles/typography.css'
import StatCardComponent from '../../components/StatCard/statcard.component'
import { useEffect } from 'react'
import { useCountries } from '../../hooks/useCountries'
import { useGDPAllocationChart, useIndicatorCards, useDistributionEnergyChart, useDistributionLandAreaChart } from '../../hooks/useIndicators'
import { useRangeYear } from '../../hooks/useRangeYear'
import PieChartComponent from '../../components/charts/PieChart/piechart.component'

const { Text } = Typography

export default function DashboardPage() {
  const { data: dataCoCo, isLoading: isLoadingCoCo } = useCountries()
  const [form] = Form.useForm()
  const selectedContinents = Form.useWatch('continent', form)
  const selectedCountries = Form.useWatch('country', form)
  const selectedPeriod = Form.useWatch('period', form)

  
  // Ràng buộc 2 select country & continent
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

  
  // Get data indicator card
  const indicatorPayload = {
    country_code_list: selectedCountries ?? [],
    fromYear: selectedPeriod?.[0],
    toYear: selectedPeriod?.[1]
  }
  const { data: dataIndicators, isFetching: isFetchingIndicators } = useIndicatorCards(indicatorPayload)

  const { data: dataRangeYear } = useRangeYear()
  let minYear = dataRangeYear?.minYear
  let maxYear = dataRangeYear?.maxYear
  const fixedMarksYear = {
    [minYear]: <span className={styleDashboard.sliderMark}>{minYear}</span>,
    [maxYear]: <span className={styleDashboard.sliderMark}>{maxYear}</span>
  }
  const filterOptionFunc = (input, option) => {
    return (option?.label ?? '').toLowerCase().includes(input.toLowerCase())
  }

  // Get data chart GDP Allocation
  const { data: dataGdpAllocation, isFetching: isFetchingGdpAllocation } = useGDPAllocationChart(indicatorPayload)
  // Get data chart Distribution Energy
  const { data: dataDistributionEnergy, isFetching: isFetchingDistributionEnergy } = useDistributionEnergyChart(indicatorPayload)
  // Get data chart Distribution Land Area
  const { data: dataDistributionLandArea, isFetching: isFetchingDistributionLandArea } = useDistributionLandAreaChart(indicatorPayload)

  return (
    <Space orientation='vertical'>
      <Form 
        layout='vertical' 
        form={form}>
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
            <Form.Item name="period" initialValue={[minYear, maxYear]}>
              <Slider
                range
                min={minYear}
                max={maxYear}
                step={1}
                marks={fixedMarksYear}
              />
            </Form.Item>
          </Col>
        </Row>
      </Form>
      <Flex wrap gap="small" align='middle' justify='center'>
        {(isFetchingIndicators ? Array.from({ length: 9 }) : dataIndicators)?.map((item, index) => (
          <div key={index} className={`${styleDashboard.statCardWrapper} ${isFetchingIndicators ? styleDashboard.loading : ''
            }`}>
            <StatCardComponent
              key={index}
              loading={isFetchingIndicators}
              icon={item?.iconMapping}
              value={item?.value}
              unitName={item?.unit}
              description={item?.description}
              calculationNote={item?.calculationNote}
            />
          </div>
        ))}
      </Flex>
      <Flex wrap gap="small" align='middle' justify='center'>
          <div className={`${styleDashboard.pieChartWrapper} ${isFetchingGdpAllocation ? styleDashboard.loading : ''
            }`}>
            <PieChartComponent data={dataGdpAllocation} titleChart='GDP Allocation by Sector' loading={isFetchingGdpAllocation}></PieChartComponent>
          </div>
          <div className={`${styleDashboard.pieChartWrapper} ${isFetchingDistributionEnergy ? styleDashboard.loading : ''
            }`}>
            <PieChartComponent data={dataDistributionEnergy} titleChart='Distribution of Energy Sources' loading={isFetchingDistributionEnergy}></PieChartComponent>
          </div>
          <div className={`${styleDashboard.pieChartWrapper} ${isFetchingDistributionLandArea ? styleDashboard.loading : ''
            }`}>
            <PieChartComponent data={dataDistributionLandArea} titleChart='Distribution of Total Land Area' loading={isFetchingDistributionLandArea}></PieChartComponent>
          </div>
      </Flex>
    </Space>
  )
}
