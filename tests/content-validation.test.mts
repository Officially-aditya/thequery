import assert from "node:assert/strict";
import test from "node:test";
import { parseContentInput } from "../lib/content-validation.ts";

test("content input stores an optional safe cover image and its alt text", () => {
  const result = parseContentInput("article", {
    title: "Cover image test",
    body: "A short article body.",
    coverImageUrl: " https://images.example.com/cover.jpg ",
    coverImageAlt: " An abstract neural network illustration ",
  });

  assert.deepEqual(result.errors, []);
  assert.equal(result.data?.coverImageUrl, "https://images.example.com/cover.jpg");
  assert.equal(result.data?.coverImageAlt, "An abstract neural network illustration");
});

test("content input rejects unsafe cover image URLs", () => {
  const result = parseContentInput("guide", {
    title: "Unsafe cover image",
    body: "A short guide body.",
    coverImageUrl: "javascript:alert(1)",
  });

  assert.equal(result.data, null);
  assert.deepEqual(result.errors, ["Cover image URL must be an http(s) URL or a site-relative path."]);
});
