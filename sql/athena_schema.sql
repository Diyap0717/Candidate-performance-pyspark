-- Conceptual AWS Athena schema from the original course project.
-- Replace the placeholder S3 location before use.

CREATE EXTERNAL TABLE IF NOT EXISTS student_performance (
    gender STRING,
    race_ethnicity STRING,
    parental_level_of_education STRING,
    lunch STRING,
    test_preparation_course STRING,
    math_score INT,
    reading_score INT,
    writing_score INT,
    final_grade STRING,
    gender_indexed DOUBLE,
    race_ethnicity_indexed DOUBLE,
    parental_level_of_education_indexed DOUBLE,
    lunch_indexed DOUBLE,
    test_preparation_course_indexed DOUBLE,
    class_label DOUBLE
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES (
    "separatorChar" = ",",
    "quoteChar" = "\""
)
STORED AS TEXTFILE
LOCATION 's3://bucket-name/student-performance-data/'
TBLPROPERTIES ("skip.header.line.count"="1");

SELECT gender, race_ethnicity, math_score, class_label
FROM student_performance
WHERE class_label = 0.0
LIMIT 10;
