import { Form, Row, Col, Typography, Select } from "antd";
import '../../styles/typography.css'
import { useCountries } from "../../hooks/useCountries";

const { Text } = Typography

export default function ForcastingPage() {
    // fixed data
    const { data: dataCountry, isLoading: isLoadingCountry } = useCountries()
    const presentYear = 2025
    const minYear = 2002
    const maxYear = presentYear + 1
    const yearOptions = Array.from(
        {length: maxYear - minYear + 1},
        (_, index) => {
            const year = minYear + index
            return {
                value: year,
                label: String(year)
            }
        }
    )
    // filter option
    const filterOptionFunc = (input, option) => {
        return (option?.label ?? '').toLowerCase().includes(input.toLowerCase())
    }
    return (
        <div>
            <Form layout="vertical">
                <Row gutter={24} style={{ marginBottom: '6px' }}>
                    <Col span={6}>
                        <Text className='label-input-filter' strong>Country</Text>
                    </Col>
                    <Col span={6}>
                        <Text className='label-input-filter' strong>Year</Text>
                    </Col>
                </Row>
                {/* form item */}
                <Row gutter={24}>
                    <Col span={6}>
                        <Form.Item name='country'>
                            <Select
                                loading={isLoadingCountry}
                                mode='single'
                                showSearch={{ filterOption: filterOptionFunc }}
                                placeholder="Select a country"
                                options={dataCountry?.map(c => ({
                                    value: c.country_code,
                                    label: c.country_name
                                }))}
                            ></Select>
                        </Form.Item>
                    </Col>
                    <Col span={6}>
                        <Form.Item name='year'>
                            <Select
                                mode='single'
                                showSearch={{ filterOption: filterOptionFunc }}
                                defaultValue={maxYear}
                                options={yearOptions}
                            ></Select>
                        </Form.Item>
                    </Col>
                </Row>
            </Form>
        </div>
    )
}