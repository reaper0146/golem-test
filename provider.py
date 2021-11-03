#!/usr/bin/env python3
import asyncio
from datetime import timedelta
import sys
from yapapi import Executor, Task, WorkContext, windows_event_loop_fix
from yapapi.log import enable_default_logger, log_summary, log_event_repr  # noqa
from yapapi.package import vm

import os
import json
from model import get_compiled_model, load_dataset, get_client_model_weights, avg_weights


GLOBAL_TRAINING_ROUNDS = 3
NUM_PROVIDERS = 3
PROVIDER_EPOCHS = 5
BATCH_SIZE = 64

SUBNET_TAG = 'devnet-beta.2'  

MODEL_WEIGHTS_FOLDER = 'output/worker_models'
WORKER_LOGS_FOLDER = 'output/logs'
ROUND_WEIGHTS_FOLDER = 'output/model_rounds'


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)


async def main():
    package = await vm.repo(
        image_hash="c0317d4db8930afde1862f27973ee2f5b766c4d50a87409406e2e23f",
        min_mem_gib=2,
        min_storage_gib=2.5,
    )

    async def worker_train_model(ctx: WorkContext, tasks):
        async for task in tasks:
            global_round = task.data['global_round']
            node_id = task.data['node_id']
            model_path = os.path.join(
                ROUND_WEIGHTS_FOLDER, f'round_{global_round - 1}.h5')
            ctx.send_file(
                model_path, f"/golem/work/model_{global_round - 1}.h5")
            using = {
                'start': task.data['start'],
                'end': task.data['end'],
                'batch_size': BATCH_SIZE,
                'model_path': f'model_{global_round - 1}.h5',
                'epochs': PROVIDER_EPOCHS,
                'global_round': task.data['global_round'],
                'node_number': task.data['node_id']
            }
            ctx.send_json(
                "/golem/work/using.json",
                using,
            )
            ctx.send_file('client.py', "/golem/work/client.py")
            ctx.run("/bin/sh", "-c", "python3 client.py")
            node_model_output = f'/golem/output/model_round_{global_round}_{node_id}.h5'
            node_log_file = f'/golem/output/log_round_{global_round}_{node_id}.json'
            ctx.download_file(node_model_output, os.path.join(
                MODEL_WEIGHTS_FOLDER, f'round_{global_round}_worker_{node_id}.h5'))
            ctx.download_file(node_log_file, os.path.join(
                WORKER_LOGS_FOLDER, f'log_round_{global_round}_worker_{node_id}.json'))
            yield ctx.commit(timeout=timedelta(minutes=7))
            task.accept_result()

    print(
        f"Initialising the model."
    )
    model = get_compiled_model()
    model.summary()
    print(
        f"Loading the data"
    )
    training_dataset, testing_dataset, train_length, test_length = load_dataset(
        BATCH_SIZE)
    print(
        f"Initial model evaluation - "
    )
    eval_results = model.evaluate(testing_dataset)
    print(
        f"ROUND 0 | Loss: {eval_results[0]} | Accuracy: {eval_results[1]}"
    )
    print(
        f"Saving Model Weights for round 0"
    )
    model.save(os.path.join(ROUND_WEIGHTS_FOLDER, 'round_0.h5'))

    for global_round_number in range(1, GLOBAL_TRAINING_ROUNDS + 1):
        print(
            f"Beginning Training Round {global_round_number}"
        )
        async with Executor(
            package=package,
            max_workers=NUM_PROVIDERS,
            budget=20.0,
            timeout=timedelta(minutes=29),
            subnet_tag=SUBNET_TAG,
            event_consumer=log_summary(log_event_repr),
        ) as executor:

            training_subset_steps = int(train_length / NUM_PROVIDERS)
            executor_tasks = [Task(data={'start': x,
                                         'end': x + training_subset_steps,
                                         'global_round': global_round_number,
                                         'node_id': index+1})
                              for index, x in enumerate(list(
                                  range(0, train_length, training_subset_steps)))]
            async for task in executor.submit(
                worker_train_model, executor_tasks
            ):
                print(
                    f"Training round {global_round_number} completed on provider node {task.data['node_id']}"
                )

        all_worker_weights = get_client_model_weights(
            MODEL_WEIGHTS_FOLDER, global_round_number)
        averaged_weights = avg_weights(all_worker_weights)
        model.set_weights(averaged_weights)

        print(
            f"TRAINING ROUND {global_round_number} complete!"
        )
        eval_results = model.evaluate(testing_dataset)
        print(
            f"ROUND {global_round_number} | Loss: {eval_results[0]} | Accuracy: {eval_results[1]}"
        )
        print(
            f"Saving Model Weights for round {global_round_number}"
        )
        model.save(os.path.join(ROUND_WEIGHTS_FOLDER,
                                f'round_{global_round_number}.h5'))
    print(
        f"TRAINING COMPLETE! FIND YOUR FINAL MODEL BY THE NAME OF"
        f" 'round_{global_round_number}.h5' IN THE OUTPUT FILES"
    )


if __name__ == "__main__":
    create_folder('output')
    create_folder(MODEL_WEIGHTS_FOLDER)
    create_folder(WORKER_LOGS_FOLDER)
    create_folder(ROUND_WEIGHTS_FOLDER)

    enable_default_logger(log_file='golemML.log')

    loop = asyncio.get_event_loop()
    task = loop.create_task(main())

    try:
        loop.run_until_complete(task)
    except KeyboardInterrupt:
        print(
            "Shutting down gracefully, please wait a short while "
            "or press Ctrl+C to exit immediately..."
        )
        task.cancel()
        try:
            loop.run_until_complete(task)
            print(
                "Shutdown completed, thank you for waiting!"
            )
        except KeyboardInterrupt:
            pass
