CREATE TABLE `missing_info` (
  `id` INT NOT NULL,
  `name` VARCHAR(200) NULL,
  `age` INT NULL,
  `address` VARCHAR(200) NULL,
  `gender` VARCHAR(45) NULL,
  PRIMARY KEY (`id`));

CREATE TABLE `images_info` (
  `id` INT NOT NULL,
  `image_name` VARCHAR(200) NULL);