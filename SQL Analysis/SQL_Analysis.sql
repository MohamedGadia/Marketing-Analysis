SELECT
*
FROM dbo.products;

--*************************************************************

-- SQL Query to categorize products based on their price
SELECT 
    ProductID,  -- Selects the unique identifier for each product
    ProductName,  -- Selects the name of each product
    Price,  -- Selects the price of each product
	-- Category, -- Selects the product category for each product

    CASE -- Categorizes the products into price categories: Low, Medium, or High
        WHEN Price < 50 THEN 'Low'  -- If the price is less than 50, categorize as 'Low'
        WHEN Price BETWEEN 50 AND 200 THEN 'Medium'  -- If the price is between 50 and 200 (inclusive), categorize as 'Medium'
        ELSE 'High'  -- If the price is greater than 200, categorize as 'High'
    END AS PriceCategory  -- Names the new column as PriceCategory

FROM 
    dbo.products;  -- Specifies the source table from which to select the data

--*************************************************************

SELECT
*
FROM dbo.customers;

SELECT
*
FROM dbo.geography;

-- SQL statement to join dim_customers with dim_geography to enrich customer data with geographic information
SELECT 
    c.CustomerID,  -- Selects the unique identifier for each customer
    c.CustomerName,  -- Selects the name of each customer
    c.Email,  -- Selects the email of each customer
    c.Gender,  -- Selects the gender of each customer
    c.Age,  -- Selects the age of each customer
    g.Country,  -- Selects the country from the geography table to enrich customer data
    g.City  -- Selects the city from the geography table to enrich customer data
FROM 
    dbo.customers as c  -- Specifies the alias 'c' for the dim_customers table
LEFT JOIN
-- RIGHT JOIN
-- INNER JOIN
-- FULL OUTER JOIN
    dbo.geography as g  -- Specifies the alias 'g' for the dim_geography table
ON 
    c.GeographyID = g.GeographyID;  -- Joins the two tables on the GeographyID field to match customers with their geographic information

--*************************************************************

SELECT
*
FROM dbo.customer_reviews;

-- Query to clean whitespace issues in the ReviewText column
SELECT 
    ReviewID,  -- Selects the unique identifier for each review
    CustomerID,  -- Selects the unique identifier for each customer
    ProductID,  -- Selects the unique identifier for each product
    ReviewDate,  -- Selects the date when the review was written
    Rating,  -- Selects the numerical rating given by the customer (e.g., 1 to 5 stars)
    -- Cleans up the ReviewText by replacing double spaces with single spaces to ensure the text is more readable and standardized
    REPLACE(ReviewText, '  ', ' ') AS ReviewText
FROM 
    dbo.customer_reviews;  -- Specifies the source table from which to select the data

--*************************************************************
SELECT
*
FROM dbo.engagement_data;

-- Query to clean and normalize the engagement_data table
SELECT 
    EngagementID,  -- Selects the unique identifier for each engagement record
    ContentID,  -- Selects the unique identifier for each piece of content
	CampaignID,  -- Selects the unique identifier for each marketing campaign
    ProductID,  -- Selects the unique identifier for each product
    UPPER(REPLACE(REPLACE(ContentType, 'Socialmedia', 'Social Media'), 'NEWSLETTER', 'NEWS LETTER')) AS ContentType,  -- Replaces "Socialmedia" with "Social Media", "NEWSLETTER" with "NEWS LETTER" and then converts all ContentType values to uppercase
    LEFT(ViewsClicksCombined, CHARINDEX('-', ViewsClicksCombined) - 1) AS Views,  -- Extracts the Views part from the ViewsClicksCombined column by taking the substring before the '-' character
    RIGHT(ViewsClicksCombined, LEN(ViewsClicksCombined) - CHARINDEX('-', ViewsClicksCombined)) AS Clicks,  -- Extracts the Clicks part from the ViewsClicksCombined column by taking the substring after the '-' character
    Likes,  -- Selects the number of likes the content received
    FORMAT(CONVERT(DATE, EngagementDate), 'dd.MM.yyyy') AS EngagementDate  -- Converts and formats the date as dd.mm.yyyy
FROM 
    dbo.engagement_data;  -- Specifies the source table from which to select the data

--*************************************************************

SELECT
*
FROM dbo.customer_journey;

-- Common Table Expression (CTE) to identify and tag duplicate records
WITH DuplicateRecords AS (
    SELECT 
        JourneyID,  -- Select the unique identifier for each journey (and any other columns you want to include in the final result set)
        CustomerID,  -- Select the unique identifier for each customer
        ProductID,  -- Select the unique identifier for each product
        VisitDate,  -- Select the date of the visit, which helps in determining the timeline of customer interactions
        Stage,  -- Select the stage of the customer journey (e.g., Awareness, Consideration, etc.)
        Action,  -- Select the action taken by the customer (e.g., View, Click, Purchase)
        Duration,  -- Select the duration of the action or interaction
        -- Use ROW_NUMBER() to assign a unique row number to each record within the partition defined below
        ROW_NUMBER() OVER (
            -- PARTITION BY groups the rows based on the specified columns that should be unique
            PARTITION BY CustomerID, ProductID, VisitDate, Stage, Action  
            -- ORDER BY defines how to order the rows within each partition (usually by a unique identifier like JourneyID)
            ORDER BY JourneyID  
        ) AS row_num  -- This creates a new column 'row_num' that numbers each row within its partition
    FROM 
        dbo.customer_journey  -- Specifies the source table from which to select the data
)

-- Select all records from the CTE where row_num > 1, which indicates duplicate entries
    
SELECT 
*
FROM DuplicateRecords
WHERE row_num > 1  -- Filters out the first occurrence (row_num = 1) and only shows the duplicates (row_num > 1)
ORDER BY JourneyID;

--*************************************************************

-- Outer query selects the final cleaned and standardized data
WITH CTE AS (
    SELECT 
        JourneyID,  -- Unique identifier for each journey
        CustomerID,  -- Unique identifier for each customer
        ProductID,  -- Unique identifier for each product
        VisitDate,  -- Date of the visit
        UPPER(Stage) AS Stage,  -- Convert Stage to uppercase for consistency
        Action,  -- Action taken by the customer
        Duration,  -- Original value of Duration
        AVG(Duration) OVER (PARTITION BY VisitDate) AS avg_duration,  -- Average Duration for each VisitDate
        ROW_NUMBER() OVER (
            PARTITION BY CustomerID, ProductID, VisitDate, UPPER(Stage), Action
            ORDER BY JourneyID
        ) AS row_num  -- Assign a row number to each record within a partition to identify unique rows
    FROM 
        dbo.customer_journey
)
SELECT 
    JourneyID,  -- Unique identifier for each journey
    CustomerID,  -- Unique identifier for each customer
    ProductID,  -- Unique identifier for each product
    VisitDate,  -- Date of the visit
    Stage,  -- Stage (converted to uppercase)
    Action,  -- Action taken by the customer
    COALESCE(Duration, avg_duration) AS Duration  -- Replace NULL values with the average Duration for that VisitDate
FROM 
    CTE
WHERE 
    row_num = 1 -- Keeps only the first occurrence of each duplicate group identified in the subquery
ORDER BY VisitDate;