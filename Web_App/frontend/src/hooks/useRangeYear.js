import { useQuery } from '@tanstack/react-query'
import { getRangeYear } from '../services/reference.service'

export const useRangeYear = () => {
  return useQuery({
    queryKey: ['range-year'],
    queryFn: getRangeYear,
    staleTime: Infinity
  })
}
