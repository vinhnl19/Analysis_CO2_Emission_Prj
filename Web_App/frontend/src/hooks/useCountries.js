import { useQuery } from '@tanstack/react-query'
import { getCountries } from '../services/country.service'

export const useCountries = () => {
  return useQuery({
    queryKey: ['countries'],
    queryFn: getCountries,
  })
}
