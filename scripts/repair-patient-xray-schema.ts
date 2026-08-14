import { config } from "dotenv";
import postgres from "postgres";

config({ path: ".env.local" });

if (!process.env.POSTGRES_URL) {
  throw new Error("POSTGRES_URL is not configured.");
}

async function main() {
  const sql = postgres(process.env.POSTGRES_URL as string, { max: 1 });

  await sql.unsafe(
    'ALTER TABLE "PatientXrayRecord" ADD COLUMN IF NOT EXISTS "patientMetadata" json'
  );
  await sql.unsafe(
    'ALTER TABLE "PatientXrayRecord" ADD COLUMN IF NOT EXISTS "tbDecision" json'
  );

  const columns = await sql.unsafe(
    `select column_name
     from information_schema.columns
     where table_name = 'PatientXrayRecord'
       and column_name in ('patientMetadata', 'tbDecision')
     order by column_name`
  );

  console.log(JSON.stringify(columns, null, 2));
  await sql.end();
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
