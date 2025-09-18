import type { Metadata } from "next";
import "./globals.css";


export const metadata: Metadata = {
  title: "Website Content Search",
  description: "Search through website content with precision",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`font-satoshi antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
