from pyfacy import face_clust
# Create object for Cluster class with your source path(only contains jpg images)
mdl = face_clust.Face_Clust_Algorithm('dataset/')
# Load the faces to the algorithm
mdl.load_faces()
# Save the group of images to custom location(if the arg is empty store to current location)
mdl.save_faces('pyfacy')
