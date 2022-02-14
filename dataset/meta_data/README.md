# Additional Meta Data 

We use the pre-processed and cleaned version of Visual Genome by [Hudson and Manning](https://arxiv.org/pdf/1902.09506.pdf). Additional metadata about each image can be found at https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip

The scene graph of each image looks like the following:

```json
{
    "2407890": {
        "width": 640,
        "height": 480,
        "location": "living room",
        "weather": none,
        "objects": {
            "271881": {
                "name": "chair",
                "x": 220,
                "y": 310,
                "w": 50,
                "h": 80,
                "attributes": ["brown", "wooden", "small"],
                "relations": {
                    "32452": {
                        "name": "on",
                        "object": "275312"
                    },
                    "32452": {
                        "name": "near",
                        "object": "279472"
                    }                    
                }
            }
        }
    }
}
```


See the following webpage for more details: https://cs.stanford.edu/people/dorarad/gqa/download.html
