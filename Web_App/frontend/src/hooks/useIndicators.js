import { useQuery } from '@tanstack/react-query';
import { getIndicatorCardData } from '../services/indicator.service';

export const useIndicatorCards = (countries, fromYear, toYear) => {
  return useQuery({
    queryKey: ['indicators', countries, fromYear, toYear], // key động dựa vào input
    queryFn: () => getIndicatorCardData( countries, fromYear, toYear ),
    enabled: countries.length > 0, // chỉ chạy khi có country
  });
};
