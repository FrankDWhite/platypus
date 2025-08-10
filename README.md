## readme is a work in progress

to install the correct dep versions:

```
pip3 install tensorflow==2.15.1 tensorflow-ranking tensorflow-serving-api
```

to start the mcp server:

```
uvicorn mcp_server:app --reload
```