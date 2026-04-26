import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        ink: "#e6e9ef",
        muted: "#8b93a3",
        bg: "#0b0d12",
        panel: "#141821",
        caller: "#ffd479",
        agent: "#a0e7a0",
        slot: "#c89cff",
        gate: "#ff9c8a",
        accent: "#5cc8ff",
      },
    },
  },
  plugins: [],
};

export default config;
