# Allen Brain Atlas Connectivity Measurement

In the Allen Brain Atlas methodology, "proportion of signal pixels" refers to the quantification of neural connections using fluorescent imaging. Here's what it specifically means:

## Measurement Process
1. When a tracer is injected into a source region, it travels along the axons to connected regions
2. The tracer fluoresces (glows) when imaged
3. Each brain region is imaged and divided into pixels
4. The "signal pixels" are pixels that show fluorescence above a threshold (indicating presence of traced connections)

## Calculation
The proportion is calculated as:

```math
Projection Density = \frac{Number\;of\;signal\;pixels}{Total\;pixels\;in\;target\;region}