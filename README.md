# How to run

Pull the docker image for Verbnet's parser and run it with the following code

```
docker pull jgung/verbnet-parser:0.1-SNAPSHOT
docker run -p 8080:8080 jgung/verbnet-parser:0.1-SNAPSHOT
```

Then, run the script

```
python verbnet.py
```
