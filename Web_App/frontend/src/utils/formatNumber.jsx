export const formatNumber = (value) => {
    if (value === null || value === undefined) return "--";

    if (typeof value !== "number") return value;

    return new Intl.NumberFormat("en-US").format(value);
};
