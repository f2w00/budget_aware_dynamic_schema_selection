CREATE TABLE "public"."artifact_raw" (
  "artifact_id" bigint NOT NULL,
  "raw_data" jsonb NOT NULL,
  PRIMARY KEY ("artifact_id")
);

CREATE TABLE "public"."artifact_extract" (
  "artifact_id" bigint NOT NULL,
  "acq_type" text,
  "category" text,
  "condition" text,
  "craft" text,
  "craft_school" text,
  "culture" text,
  "desc" text,
  "dynasty" text,
  "form_type" text,
  "material" text,
  "motif" text,
  "name" text,
  "period" text,
  "region" text,
  "related_event" text,
  "related_person" text,
  "site" text,
  "usage" text,
  PRIMARY KEY ("artifact_id")
);
