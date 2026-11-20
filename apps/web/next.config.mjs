/** @type {import('next').NextConfig} */
const nextConfig = {
  // Static export: the whole app is client-side (Zustand + EventSource against
  // the FastAPI backend), so it can be bundled and loaded from disk by Electron
  // without running a Node server at runtime.
  output: "export",
  transpilePackages: ["@ru-socrates/types"],
  experimental: {
    typedRoutes: false,
  },
};

export default nextConfig;
