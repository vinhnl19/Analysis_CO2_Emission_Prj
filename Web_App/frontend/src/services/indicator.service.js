import { apiClient } from "./apiClient";

export const getIndicatorCardData = async (countries, fromYear, toYear) => {
    const payload = {
        country_code_list: countries,
        fromYear: fromYear,
        toYear: toYear
    }
  const res = await apiClient.post("/indicator/getfromcountry", payload);
  return res.data;
};
