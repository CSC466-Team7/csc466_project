import process from "process";

export const IS_DEV: boolean = !process.env.NODE_ENV || process.env.NODE_ENV === "development";

export const ASSETS_URL = "https://raw.githubusercontent.com/CSC466-Team7/csc466_project/main";

export const MARKDOWN_ASSETS_URL = `${ASSETS_URL}/code/markdown`;
export const NOTEBOOK_ASSETS_URL = `${ASSETS_URL}/code/notebooks`;
export const DATASET_ASSETS_URL = `${ASSETS_URL}/datasets`;