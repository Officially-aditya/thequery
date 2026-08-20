import { migrate } from "./migrate.mjs";
import { seedContent } from "./seed-content.mjs";

await migrate();
await seedContent();
