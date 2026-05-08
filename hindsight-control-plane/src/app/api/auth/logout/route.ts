import { NextResponse } from "next/server";

const ACCESS_KEY_COOKIE = "hindsight_cp_access";

export async function POST() {
  const response = NextResponse.json({ success: true });

  response.cookies.set({
    name: ACCESS_KEY_COOKIE,
    value: "",
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    path: "/",
    maxAge: 0,
  });

  return response;
}
