
/*

It is a collection of the State Budget of Angolan government from 2002 to 2021 for Health and Education sectors.
Source: https://www.minfin.gov.ao/PortalMinfin/#!/materias-de-realce/orcamento-geral-do-estado/oge-passados

*/


/*creating the database*/

CREATE DATABASE HealthandEducationStateBudget;


/*creating and inserting the data into the tables*/


CREATE TABLE budget (
Year INT IDENTITY (2002,1) PRIMARY KEY,
GovBudg_Health INT,
GovBudg_Education INT);

		--modifying the data type--

		ALTER TABLE budget
		ALTER COLUMN GovBudg_Health REAL;

		ALTER TABLE budget
		ALTER COLUMN GovBudg_Education REAL;


INSERT INTO budget (GovBudg_Health, GovBudg_Education)
VALUES (4.57, 5.19), (5.82, 6.24), (5.69, 10.47), (4.97, 7.14), (4.42, 3.82),
(3.68, 5.61), (6.68, 7.91), (8.38, 7.9), (5.02, 8.52), (1.61, 1.04),
(5.14, 8.37), (5.56, 8.83), (4.35, 6.16), (5.05, 8.92), (4.35, 6.55),
(4.30, 6.76), (4.01, 6.49), (5.65, 6.05), (6.07, 6.47), (5.76, 6.92);


/* Converting the Budget values, given as percentage, to decimal*/

UPDATE budget set GovBudg_Health = GovBudg_Health/100;
UPDATE budget set GovBudg_Education = GovBudg_Education/100;


/*checking the data type*/

EXEC sp_help budget;


/*Displaying the database*/

SELECT * FROM budget;


/*
At this point the database was ready. It was then converted to a dataframe and concatenated with HDI table using Pandas library, from Python
*/