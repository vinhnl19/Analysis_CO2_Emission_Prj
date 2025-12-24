import { useQuery } from '@tanstack/react-query';
import { getIndicatorCardData } from '../services/indicator.service';
import { INDICATOR_TEMPLATE } from '../constants/indicatorTemplate';

export const useIndicatorCards = (payload) => {
  const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms))
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
    // enabled: Boolean(!!payload &&
    //      payload.country_code_list.length > 0 &&
    //      payload.fromYear &&
    //      payload.toYear),
  });
};
