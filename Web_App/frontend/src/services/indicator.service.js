import { apiClient } from "./apiClient";

export const getIndicatorCardData = async ({country_code_list, fromYear, toYear}) => {
    const payload = {
        country_code_list: country_code_list,
        fromYear: fromYear,
        toYear: toYear
    }
  const res = await apiClient.post("/indicator/getfromcountry", payload);
  return res.data;
};

export const getGdpAllocationChartData = async ({country_code_list, fromYear, toYear}) => {
  const payload = {
    country_code_list: country_code_list,
    fromYear: fromYear,
    toYear: toYear
  }
  const res = await apiClient.post("/indicator/get-gdp-allocation", payload)
  return res.data
}
export const getDistributionEnergyChartData = async ({country_code_list, fromYear, toYear}) => {
  const payload = {
    country_code_list: country_code_list,
    fromYear: fromYear,
    toYear: toYear
  }
  const res = await apiClient.post("/indicator/get-distribution-energy", payload)
  return res.data
}
export const getDistributionLandAreaChartData = async ({country_code_list, fromYear, toYear}) => {
  const payload = {
    country_code_list: country_code_list,
    fromYear: fromYear,
    toYear: toYear
  }
  const res = await apiClient.post("/indicator/get-distribution-land-area", payload)
  return res.data
}
