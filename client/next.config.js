/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  eslint: {
    ignoreDuringBuilds: true,
  },
  images: { 
    unoptimized: true,
    remotePatterns: [
      {
        protocol: 'https',
        hostname: '**',
      },
      {
        protocol: 'http',
        hostname: '**',
      },
    ],
  },
  // async headers() {
  //   return [
  //     {
  //       source: '/:path*',
  //       headers: [
  //         {
  //           key: 'Access-Control-Allow-Origin',
  //           value: 'http://localhost:3000',
  //         },
  //         {
  //           key: 'Access-Control-Allow-Methods',
  //           value: 'GET,OPTIONS,PATCH,DELETE,POST,PUT',
  //         },
  //         {
  //           key: 'Access-Control-Allow-Headers',
  //           value: 'X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version, Authorization',
  //         },
  //       ],
  //     },
  //   ];
  // },
};

module.exports = nextConfig;
