# MasterThesis: A Geoguessr based approach to location determination

### Region guessing within a country:
- This is depended on the hints found and country:
  - For example: Belgium(https://www.plonkit.net/belgium) can generated other hints than the Netherlands(https://www.plonkit.net/netherlands, https://docs.google.com/document/d/1oddLtz2lFn8F43fsie2GI4-FvOFZAVNNM4KvrJelWwQ/edit?tab=t.0#heading=h.906694vnlz5w)  
   
for region guessing and guessing in general but do we have to make this specific for each country or in general.

A) Make it general:
- This leads to need of 1 NN with similarly trained data
- But country specific hints are lost   
  
B) Make it Country specific:
- This leads to the need of as many NN as suported countries 
- Allow each NN to learn specific country hints like local phone numbers, city names, and others...
- Disadvantage: these hints are mostly gather from geogussr tip boards  
  

Overall model Geo-Clip:
- Paper: https://arxiv.org/pdf/2309.16020 
- Github: https://github.com/VicenteVivan/geo-clip 