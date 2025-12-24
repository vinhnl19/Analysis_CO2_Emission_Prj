import { apiClient } from "./apiClient";

export const getRangeYear = async () => {
  const res = await apiClient.get("/reference/rangeyear");
  return res.data;
};
