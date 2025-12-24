import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { Router, RouterProvider } from 'react-router-dom'
import { router } from "./router.jsx"
import { ConfigProvider } from 'antd'
import { setTwoToneColor } from '@ant-design/icons';
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import './index.css'
import 'antd/dist/reset.css'

setTwoToneColor(['#033e0c', '#d3f3d8'])

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    }
  }
})

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <ConfigProvider
        theme={{
          token: {
            fontFamily: 'Inter, system-ui, Avenir, Helvetica, Arial, sans-serif',
            colorPrimary: "#668f6b"
          },
          components: {
            Select: {
              colorBorder: "#668f6b", // màu đường viền
              colorPrimaryHover: "#d3f3d8", // màu đường viền khi hover
              colorPrimary: "#033e0c",
              optionSelectedBg: "#d3f3d8", // màu bg của option được chọn 
              optionSelectedColor: "#033e0c", // màu chữ của option được chọn
              controlOutline: "#d3f3d8",
              multipleItemBg: "#d3f3d8",
              optionActiveBg: '#ecfeee',
              colorTextPlaceholder: "#668f6b89"
            },
            Slider: {
              trackBg: '#668f6b',          // đoạn được chọn
              trackHoverBg: '#668f6b',

              railBg: '#d1dfd3',           // đoạn chưa chọn
              railHoverBg: '#d1dfd3',

              handleColor: '#668f6b',      // viền núm kéo
              handleActiveColor: '#668f6b',
              handleActiveOutlineColor: '#668f6b',

              dotBorderColor: '#d1dfd3',
              dotActiveBorderColor: '#668f6b',
            },
            Card: {
              borderRadiusLG: '16px',
              colorBgContainer: 'rgb(178, 233, 187, 0.3)',
              paddingLG: '16px',
              paddingSM: '12px'
            },
            Tooltip: {
              colorBgSpotlight: "#1f4f3a",     // nền tooltip
              colorTextLightSolid: "#ecfeee",  // chữ
              borderRadius: 8,
              paddingSM: 12,
              fontSize: 13,
              lineHeight: 1.5,
            },
            Popover: {
              colorBgElevated: "#1f4f3a", // nền popover
              colorText: "#ecfeee",       // màu chữ
              colorTextHeading: "#ecfeee",
              fontWeightStrong: 700,
              borderRadiusLG: 12,
              boxShadow: "0 8px 24px rgba(0,0,0,0.25)"
            }
          }
        }}
      >
        <RouterProvider router={router} />
      </ConfigProvider>
    </QueryClientProvider>
  </StrictMode>,
)
