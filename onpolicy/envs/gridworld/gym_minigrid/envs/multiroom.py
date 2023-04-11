from onpolicy.envs.gridworld.gym_minigrid.minigrid import *
from onpolicy.envs.gridworld.gym_minigrid.register import register
from .irregular_room import *

class Multi_Room:
    def __init__(self,
        top,
        size,
        entryDoorPos_1,
        entryDoorPos_2,
        entryDoorPos_3,
        entryDoorPos_4,
        exitDoorPos_1,
        exitDoorPos_2,
        exitDoorPos_3,
        exitDoorPos_4,
    ):
        self.top = top
        self.size = size
        self.entryDoorPos_1 = entryDoorPos_1
        self.entryDoorPos_2 = entryDoorPos_2
        self.entryDoorPos_3 = entryDoorPos_3
        self.entryDoorPos_4 = entryDoorPos_4
        self.exitDoorPos_1 = exitDoorPos_1
        self.exitDoorPos_2 = exitDoorPos_2
        self.exitDoorPos_3 = exitDoorPos_3
        self.exitDoorPos_4 = exitDoorPos_4

class MultiRoomEnv(Irregular_RoomEnv):
    """
    Environment with multiple rooms (subgoals)
    """

    def __init__(self,
        grid_size, 
        max_steps, 
        num_agents, 
        agent_view_size, 
        use_merge,
        use_merge_plan,
        use_constrict_map,
        use_fc_net,
        use_agent_id,
        use_stack,
        use_orientation,
        minNumRooms,
        maxNumRooms,
        maxRoomSize=10     
    ):
        assert minNumRooms > 0
        assert maxNumRooms >= minNumRooms
        assert maxRoomSize >= 4

        self.minNumRooms = minNumRooms
        self.maxNumRooms = maxNumRooms
        self.maxRoomSize = maxRoomSize
        self.explorable_size = 0
        self.rooms = []

        super(MultiRoomEnv, self).__init__(
            grid_size = grid_size, 
            max_steps = max_steps, 
            num_agents = num_agents, 
            agent_view_size = agent_view_size, 
            use_merge = use_merge,
            use_merge_plan = use_merge_plan,
            use_constrict_map = use_constrict_map,
            use_fc_net = use_fc_net,
            use_agent_id = use_agent_id,
            use_stack = use_stack,
            use_orientation = use_orientation,
            minNumRooms = minNumRooms,
            maxNumRooms = maxNumRooms,
            maxRoomSize=10
        )

    def multiroom_gen_grid(self, width, height):
        roomList = []

        # Choose a random number of rooms to generate
        numRooms = self._rand_int(self.minNumRooms, self.maxNumRooms+1)

        while len(roomList) < numRooms:
            curRoomList = []

            entryDoorPos_1 = (
                self._rand_int(0, width - 2),
                self._rand_int(0, width - 2)
            )

            entryDoorPos_2 = (
                self._rand_int(0, width - 2),
                self._rand_int(0, width - 2)
            )

            entryDoorPos_3 = (
                self._rand_int(0, width - 2),
                self._rand_int(0, width - 2)
            )

            entryDoorPos_4 = (
                self._rand_int(0, width - 2),
                self._rand_int(0, width - 2)
            )

            # Recursively place the rooms
            self._place_Multi_Room(
                numRooms,
                roomList=curRoomList,
                minSz=4,
                maxSz=self.maxRoomSize,
                entryDoorWall=2,
                entryDoorPos_1=entryDoorPos_1,
                entryDoorPos_2=entryDoorPos_2,
                entryDoorPos_3=entryDoorPos_3,
                entryDoorPos_4=entryDoorPos_4,
            )

            if len(curRoomList) > len(roomList):
                roomList = curRoomList

        # Store the list of rooms in this environment
        assert len(roomList) > 0
        self.rooms = roomList

        # Create the grid
        self.grid = Grid(width, height)
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        wall = Wall()

        prevDoorColor = None

        # For each room
        for idx, room in enumerate(roomList):

            topX, topY = room.top
            sizeX, sizeY = room.size

            # Draw the top and bottom walls
            for i in range(0, sizeX):
                self.grid.set(topX + i, topY, wall)
                self.grid.set(topX + i, topY + sizeY - 1, wall)

            # Draw the left and right walls
            for j in range(0, sizeY):
                self.grid.set(topX, topY + j, wall)
                self.grid.set(topX + sizeX - 1, topY + j, wall)

            # If this isn't the first room, place the entry door
            if idx > 0:
                # # Pick a door color different from the previous one
                # doorColors = set(COLOR_NAMES)
                # if prevDoorColor:
                #     doorColors.remove(prevDoorColor)
                # # Note: the use of sorting here guarantees determinism,
                # # This is needed because Python's set is not deterministic
                # doorColor = self._rand_elem(sorted(doorColors))

                # entryDoor = Door(doorColor)
                self.grid.set(*room.entryDoorPos_1, None)
                self.grid.set(*room.entryDoorPos_2, None)
                self.grid.set(*room.entryDoorPos_3, None)
                self.grid.set(*room.entryDoorPos_4, None)
                # prevDoorColor = doorColor

                prevRoom = roomList[idx-1]
                prevRoom.exitDoorPos_1 = room.entryDoorPos_1
                prevRoom.exitDoorPos_2 = room.entryDoorPos_2
                prevRoom.exitDoorPos_3 = room.entryDoorPos_3
                prevRoom.exitDoorPos_4 = room.entryDoorPos_4

        # Randomize the starting agent position and direction
        num = self._rand_int(0, len(roomList))
        self.place_agent(roomList[num].top, roomList[num].size, use_same_location = self.use_same_location)

        # Place the final goal in the last room
        #self.goal_pos = self.place_obj(Goal(), roomList[-1].top, roomList[-1].size)
        for i in range(numRooms):
            self.explorable_size += roomList[i].size[0] * roomList[i].size[1]
        self.mission = 'traverse the rooms to get to the goal'

    def _place_Multi_Room(
        self,
        numLeft,
        roomList,
        minSz,
        maxSz,
        entryDoorWall,
        entryDoorPos_1,
        entryDoorPos_2,
        entryDoorPos_3,
        entryDoorPos_4,
    ):
        # Choose the room size randomly
        sizeX = self._rand_int(minSz, maxSz+1)
        sizeY = self._rand_int(minSz, maxSz+1)

        # The first room will be at the door position
        if len(roomList) == 0:
            topX, topY = entryDoorPos_1
        # Entry on the right
        elif entryDoorWall == 0:
            topX = entryDoorPos_3[0] - sizeX + 1
            y = entryDoorPos_3[1]
            topY = self._rand_int(y - sizeY + 2, y)
        # Entry wall on the south
        elif entryDoorWall == 1:
            x = entryDoorPos_4[0]
            topX = self._rand_int(x - sizeX + 2, x)
            topY = entryDoorPos_4[1] - sizeY + 1
        # Entry wall on the left
        elif entryDoorWall == 2:
            topX = entryDoorPos_1[0]
            y = entryDoorPos_1[1]
            topY = self._rand_int(y - sizeY + 2, y)
        # Entry wall on the top
        elif entryDoorWall == 3:
            x = entryDoorPos_2[0]
            topX = self._rand_int(x - sizeX + 2, x)
            topY = entryDoorPos_2[1]
        else:
            assert False, entryDoorWall

        # If the room is out of the grid, can't place a room here
        if topX < 2 or topY < 2:
            return False
        if topX + sizeX > self.width-2 or topY + sizeY >= self.height-2:
            return False

        # If the room intersects with previous rooms, can't place it here
        for room in roomList[:-1]:
            nonOverlap = \
                topX + sizeX < room.top[0] or \
                room.top[0] + room.size[0] <= topX or \
                topY + sizeY < room.top[1] or \
                room.top[1] + room.size[1] <= topY

            if not nonOverlap:
                return False

        # Add this room to the list
        roomList.append(Multi_Room(
            (topX, topY),
            (sizeX, sizeY),
            entryDoorPos_1,
            entryDoorPos_2,
            entryDoorPos_3,
            entryDoorPos_4,
            None,
            None,
            None,
            None
        ))

        # If this was the last room, stop
        if numLeft == 1:
            return True

        # Try placing the next room
        for i in range(0, 8):

            # Pick which wall to place the out door on
            wallSet = set((0, 1, 2, 3))
            wallSet.remove(entryDoorWall)
            exitDoorWall = self._rand_elem(sorted(wallSet))
            nextEntryWall = (exitDoorWall + 2) % 4

            # Pick the exit door position
            # Exit on right wall
            #if exitDoorWall == 0:
            exitDoorPos_1 = (
                topX + sizeX - 1,
                topY + self._rand_int(1, sizeY - 1)
            )
            # Exit on south wall
            #elif exitDoorWall == 1:
            exitDoorPos_2 = (
                topX + self._rand_int(1, sizeX - 1),
                topY + sizeY - 1
            )
            # Exit on left wall
            #elif exitDoorWall == 2:
            exitDoorPos_3 = (
                topX,
                topY + self._rand_int(1, sizeY - 1)
            )
            # Exit on north wall
            #elif exitDoorWall == 3:
            exitDoorPos_4 = (
                topX + self._rand_int(1, sizeX - 1),
                topY
            )
            #else:
                #assert False

            # Recursively create the other rooms
            success = self._place_Multi_Room(
                numLeft - 1,
                roomList=roomList,
                minSz=minSz,
                maxSz=maxSz,
                entryDoorWall=nextEntryWall,
                entryDoorPos_1=exitDoorPos_1,
                entryDoorPos_2=exitDoorPos_2,
                entryDoorPos_3=exitDoorPos_3,
                entryDoorPos_4=exitDoorPos_4,
            )

            if success:
                break

        return True

class MultiRoomEnvN2S4(MultiRoomEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=2,
            maxNumRooms=2,
            maxRoomSize=4
        )

class MultiRoomEnvN4S5(MultiRoomEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=4,
            maxNumRooms=4,
            maxRoomSize=5
        )

class MultiRoomEnvN6(MultiRoomEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=6,
            maxNumRooms=6
        )

'''register(
    id='MiniGrid-MultiRoom-N2-S4-v0',
    entry_point='onpolicy.envs.gridworld.gym_minigrid.envs:MultiRoomEnvN2S4'
)

register(
    id='MiniGrid-MultiRoom-N4-S5-v0',
    entry_point='onpolicy.envs.gridworld.gym_minigrid.envs:MultiRoomEnvN4S5'
)

register(
    id='MiniGrid-MultiRoom-N6-v0',
    entry_point='onpolicy.envs.gridworld.gym_minigrid.envs:MultiRoomEnvN6'
)'''
