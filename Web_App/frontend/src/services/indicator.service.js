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
