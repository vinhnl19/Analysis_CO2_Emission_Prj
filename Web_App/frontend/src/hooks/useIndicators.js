import { useQuery } from '@tanstack/react-query';
import { getGdpAllocationChartData, getIndicatorCardData, getDistributionEnergyChartData, getDistributionLandAreaChartData } from '../services/indicator.service';
import { INDICATOR_TEMPLATE } from '../constants/indicatorTemplate';

const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms))
export const useIndicatorCards = (payload) => {
  return useQuery({
    queryKey: ['indicators', payload], // key động dựa vào input
    queryFn: async () => {
      if (
        !payload ||
        payload.country_code_list?.length === 0 ||
        !payload.fromYear ||
        !payload.toYear
      ) {
        await sleep(300)
        return INDICATOR_TEMPLATE
      }
      await sleep(300)

      return getIndicatorCardData(payload)
    },
    keepPreviousData: true,
  });
};

export const useGDPAllocationChart = (payload) => {
  return useQuery({
    queryKey: ['gdp_allocation', payload],
    queryFn: async () => {
      if (
        !payload ||
        payload.country_code_list?.length === 0 ||
        !payload.fromYear ||
        !payload.toYear
      ) {
        await sleep(300)
        return null
      }
      await sleep(300)

      return  getGdpAllocationChartData(payload)
    },
    keepPreviousData: true
  })
}
export const useDistributionEnergyChart = (payload) => {
  return useQuery({
    queryKey: ['distribution_energy', payload],
    queryFn: async () => {
      if (
        !payload ||
        payload.country_code_list?.length === 0 ||
        !payload.fromYear ||
        !payload.toYear
      ) {
        await sleep(300)
        return null
      }
      await sleep(300)

      return  getDistributionEnergyChartData(payload)
    },
    keepPreviousData: true
  })
}
export const useDistributionLandAreaChart = (payload) => {
  return useQuery({
    queryKey: ['distribution_land_area', payload],
    queryFn: async () => {
      if (
        !payload ||
        payload.country_code_list?.length === 0 ||
        !payload.fromYear ||
        !payload.toYear
      ) {
        await sleep(300)
        return null
      }
      await sleep(300)

      return  getDistributionLandAreaChartData(payload)
    },
    keepPreviousData: true
  })
}