import os


def save(
        saved_model,
        saved_score,
        location='/mnt/e/mlflow_experiments',
        module_name='keras_diabetes'
):
    if not os.path.exists(f"{location}/{module_name}"):
        os.mkdir(f"{location}/{module_name}")

    directories = os.listdir(f"{location}/{module_name}")
    for i in range(0, 10000):
        dirname = f"model_{i:04}"
        if dirname not in directories:
            # save model
            os.mkdir(f"{location}/{module_name}/{dirname}")
            saved_model.save(f"{location}/{module_name}/{dirname}/model.keras")

            # save score
            text_file = open(f"{location}/{module_name}/{dirname}/score.txt", "w")
            text_file.write(str(saved_score))
            text_file.close()

            # end the loop
            break
