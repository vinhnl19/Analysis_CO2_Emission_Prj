import { apiClient } from "./apiClient";

export const getCountries = async () => {
  const res = await apiClient.get("/country/getall");
  return res.data;
};
