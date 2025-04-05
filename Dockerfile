# Run dockerized dipnet player or advisor bot
FROM python:3.7

WORKDIR /app
COPY . /app

RUN apt-get update && apt-get install -y g++ libstdc++6
RUN cd /app
RUN pip install -r requirements.txt
RUN g++ -std=c++11 -shared diplomacy_research/models/user_ops/seeded_random.cc -o seeded_random.so -fPIC $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') -O2

ENTRYPOINT ["python", "-m", "diplomacy_research.bots.run_bot"]
CMD ["--host", "localhost", "--port", "8000", "--game_id", "dn_test", "--power", "GERMANY", "--bot_type", "DipnetPlayer"]
    