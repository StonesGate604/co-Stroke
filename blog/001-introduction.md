# co-Stroke: a human-AI drawing process experiment

This is the first development note for **co-Stroke**, a personal project about drawing, time, and human-AI collaboration.

co-Stroke grows out of my AI Drawing Studio project from Code Your Way at NYU ITP. In that earlier project, I built a drawing interface that could record user strokes, export drawing sessions, replay the process on a timeline, and begin experimenting with AI-generated continuation. That phase helped me realize that the most interesting part of the project was not simply whether AI could generate an image. The real question was whether AI could participate in the process of drawing.

Most generative image tools focus on the final result. A user writes a prompt, the model returns an image, and the creative process between those two points is mostly hidden. co-Stroke starts from a different assumption: a drawing is not only an image, but also a sequence of actions over time. Every stroke has an order, a direction, a rhythm, and a relationship to the strokes before it.

The new goal is to treat drawing as a stroke sequence. Each stroke can be understood as a token, and the next token can be predicted from the previous tokens. This makes the project closer to an autoregressive language model than to a conventional image generator. Instead of predicting pixels, co-Stroke will try to predict drawing actions.

This change also responds to a practical problem in my previous approach. I originally tried to collect user drawing data through my own web platform, but that plan was not realistic. Data collection was slow, users did not always know what to draw, and the resulting data would likely contain too much noise. For the next phase, I plan to use the Quick, Draw! dataset as the main training source because it already contains a large amount of stroke-based sketch data.

The first version of co-Stroke will focus on simple line drawings, similar in spirit to SketchRNN. I am intentionally narrowing the goal. I am not trying to build a model that colors complex sketches or generates polished images. I want to build a system where a human and an AI can work on the same stroke timeline: the AI can draw, the human can pause it, continue from any point, branch the drawing process, and let the AI continue again from the new history.

In the long term, I also want to connect this system to a robot arm. The stroke sequence created by the human and AI would not only exist on screen, but could be translated into physical movement and drawn on paper. This would turn co-Stroke into a system that connects digital generation, human intervention, and physical execution.

The project will develop in several stages:

1. Define a stroke sequence data format.
2. Build a timeline player that can replay stroke data.
3. Add human continuation inside the timeline.
4. Convert Quick, Draw! data into the co-Stroke format.
5. Train a small autoregressive stroke model.
6. Connect the model to the web interface.
7. Convert stroke sequences into robot-arm drawing paths.

The central question behind co-Stroke is simple:

**If every stroke can be recorded, replayed, predicted, and continued, can drawing become a shared sequence between a human, an AI model, and a machine?**
